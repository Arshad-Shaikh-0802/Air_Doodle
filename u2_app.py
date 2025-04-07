# $env:GOOGLE_APPLICATION_CREDENTIALS = "D:/Sem_6/ML/credentials.json"

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import io
import os
from PIL import Image
from google.cloud import vision
import mediapipe as mp
import threading
from ultralytics import YOLO

from tensorflow.keras.models import load_model

# Load Quick, Draw! object recognition model
model = load_model("keras.h5")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

#YOLO Model
model_yolo = YOLO("weights/best.pt")
print("YOLO model loaded with classes:", model_yolo.names)

app = Flask(__name__)
mode = "text"  # default


# Google Vision Client
client = vision.ImageAnnotatorClient()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Drawing Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0

# Drawing Settings
colors = [(255,255,255),(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0
draw_color = colors[color_index]
line_thickness = 15
canvas_opacity = 0.3
recognized_text = ""

# Threaded Video Capture
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        global prev_x, prev_y, canvas, draw_color
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    x1, y1 = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    x2, y2 = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])

                    if np.hypot(x1 - x2, y1 - y2) > 60:
                        if prev_x and prev_y:
                            cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, line_thickness)
                        prev_x, prev_y = x1, y1
                    else:
                        prev_x, prev_y = 0, 0

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            combined = cv2.addWeighted(frame, 1 - canvas_opacity, canvas, canvas_opacity, 0)
            self.frame = combined

    def get_frame(self):
        if self.frame is not None:
            _, buffer = cv2.imencode('.jpg', self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buffer.tobytes()
        return None

    def stop(self):
        self.running = False
        self.cap.release()

camera = VideoCamera()

# OCR Preprocessing

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.dilate(thresh, kernel, iterations=1)
    return cleaned

# OCR

def recognize_text_google(image_np):
    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    image = vision.Image(content=buffer.getvalue())
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""


# Object Preprocessing for CNN

def preprocess_canvas_for_model(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = thresh[y:y+h, x:x+w]

    size = 20
    if h > w:
        new_h = size
        new_w = int(w * (size / h))
    else:
        new_w = size
        new_h = int(h * (size / w))
    roi_resized = cv2.resize(roi, (new_w, new_h))

    canvas28 = np.full((28, 28), 0, dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    canvas28 = canvas28.astype("float32") / 255.0
    return canvas28.reshape(1, 28, 28, 1)

# Object Detection by CNN
def predict_drawing(canvas):
    input_img = preprocess_canvas_for_model(canvas)
    prediction = model.predict(input_img, verbose=0)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index] * 100

    if confidence < 40:
        return "Uncertain", confidence
    return class_names[max_index], confidence



# Object Preprocess and Detection for YOLO

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours and get bounding box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = image[y:y+h, x:x+w]
        return cropped
    return image  # Return original if no contours found

def predict_by_YOLO(canvas):
    try:
        # Invert canvas (white drawings on black)
        inv_canvas = cv2.bitwise_not(canvas)

        # Convert to RGB and crop
        image = cv2.cvtColor(inv_canvas, cv2.COLOR_BGR2RGB)
        cropped_image = auto_crop(image)

        # Save for manual checking
        # cv2.imwrite("static/debug_input.png", cropped_image)

        # Run detection
        results = model_yolo(cropped_image)

        for result in results:
            print("🔍 YOLO result:", result)
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model_yolo.names[cls_id]
                    confidence = float(box.conf[0]) * 100
                    print("✅ Prediction:", label, confidence)
                    return label, confidence

        print("⚠️ YOLO: No detections found.")
        return "Uncertain", 0.0

    except Exception as e:
        print("❌ YOLO Error:", e)
        return "Error", 0.0


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize', methods=['POST'])
def recognize():
    global recognized_text
    if mode == "text":
        processed = preprocess_for_ocr(canvas)
        recognized_text = recognize_text_google(processed)
        return jsonify({"text": recognized_text})
    elif mode == "object":
        label, confidence = predict_drawing(canvas)
        return jsonify({"label": label, "confidence": f"{confidence:.2f}"})
    elif mode == "object_yolo":
        label, confidence = predict_by_YOLO(canvas)
        return jsonify({"label": label, "confidence": f"{confidence:.2f}"})

@app.route('/color', methods=['POST'])
def change_color():
    global color_index, draw_color
    color_index = (color_index + 1) % len(colors)
    draw_color = colors[color_index]
    return ('', 204)

@app.route('/clear', methods=['POST'])
def clear():
    global canvas, recognized_text, prev_x, prev_y
    canvas[:] = 0
    recognized_text = ""
    prev_x, prev_y = 0, 0
    return ('', 204)

@app.route('/save', methods=['POST'])
def save():
    mode = request.json.get("mode")
    filename = ""
    if mode == 'with_bg':
        frame = camera.frame if camera.frame is not None else canvas.copy()
        output_img = cv2.addWeighted(frame, 1 - canvas_opacity, canvas, canvas_opacity, 0)
        filename = "airwriting_with_background.png"
        cv2.imwrite(os.path.join("static", filename), output_img)
    elif mode == 'transparent':
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(canvas)
        rgba = cv2.merge((b, g, r, alpha))
        filename = "airwriting_transparent.png"
        cv2.imwrite(os.path.join("static", filename), rgba)
    return jsonify({"filename": filename})

# @app.route('/recognize_object', methods=['POST'])
# def recognize_object():
#     label, confidence = predict_drawing(canvas)
#     return jsonify({"label": label, "confidence": f"{confidence:.2f}"})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global mode
    selected_mode = request.json.get("mode")
    if selected_mode in ["text", "object", "object_yolo"]:
        mode = selected_mode
    return ('', 204)


@app.route('/shutdown')
def shutdown():
    camera.stop()
    return "Camera stopped."


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=False, threaded=True)

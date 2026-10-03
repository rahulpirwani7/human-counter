from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np

from pathlib import Path
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "yolo26s_saved_model"

print("Loading YOLO model...")

model = tf.saved_model.load(str(MODEL_PATH))
predict = model.signatures["serving_default"]

print("YOLO model loaded successfully")


@app.route("/")
def index():
    return "Human Counter backend"


@app.route("/detect_yolo", methods=["POST"])
def detect_yolo():

    if "image" not in request.files:
        return jsonify({
            "error": "No image received"
        }), 400

    image_file = request.files["image"]

    try:
        img = Image.open(image_file).convert("RGB")

        resized_img = img.resize((640, 640))

        input_array = np.array(
            resized_img,
            dtype=np.float32
        ) / 255.0

        input_tensor = tf.convert_to_tensor(
            input_array[np.newaxis, ...]
        )

        output = predict(input_tensor)

        predictions = output["output_0"][0].numpy()

        candidates = predictions.transpose()

        boxes = candidates[:, :4]
        scores = candidates[:, 4]

        confidence_threshold = 0.40

        mask = scores >= confidence_threshold

        boxes = boxes[mask]
        scores = scores[mask]

        center_x = boxes[:, 0]
        center_y = boxes[:, 1]

        width = boxes[:, 2]
        height = boxes[:, 3]

        left_edge = center_x - width / 2
        right_edge = center_x + width / 2

        top_edge = center_y - height / 2
        bottom_edge = center_y + height / 2

        nms_boxes = np.stack(
            (
                top_edge,
                left_edge,
                bottom_edge,
                right_edge
            ),
            axis=1
        )

        selected_indices = tf.image.non_max_suppression(
            nms_boxes,
            scores,
            max_output_size=100,
            iou_threshold=0.5
        )

        final_boxes = tf.gather(
            nms_boxes,
            selected_indices
        ).numpy()

        people_count = len(final_boxes)

        draw = ImageDraw.Draw(resized_img)

        for top, left, bottom, right in final_boxes:

            left = max(0, min(639, float(left)))
            right = max(0, min(639, float(right)))

            top = max(0, min(639, float(top)))
            bottom = max(0, min(639, float(bottom)))

            draw.rectangle(
                [
                    left,
                    top,
                    right,
                    bottom
                ],
                outline="red",
                width=3
            )

        image_buffer = BytesIO()

        resized_img.save(
            image_buffer,
            format="JPEG"
        )

        image_buffer.seek(0)

        encoded_image = base64.b64encode(
            image_buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "count": people_count,
            "image": "data:image/jpeg;base64," + encoded_image
        })

    except Exception as error:

        print("Detection error:", error)

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000,
        use_reloader=False
    )
from PIL import Image,ImageDraw
import numpy as np
import tensorflow as tf

img = Image.open("test_images/test2.png")
print(img.size)
print(img.mode)

new_img = img.convert("RGB")
print(new_img.mode)

np_img=np.array(new_img)
print(np_img.shape)

model = tf.saved_model.load("model/saved_model")
print("Model loaded successfully")

model_input = tf.convert_to_tensor(np_img)
input_tensor = tf.expand_dims(model_input,axis=0)

detections = model(input_tensor)
print(detections.keys())
score =detections['detection_scores'][0]
classes=detections['detection_classes'][0]
boxes=detections['detection_boxes'][0]


person_mask = (classes == 1) & (score >= 0.45)

person_count = tf.reduce_sum(tf.cast(person_mask, tf.int32)).numpy()
print(f"Number of persons detected: {person_count}")
print(boxes[person_mask].numpy())

draw=ImageDraw.Draw(new_img)

for box in boxes[person_mask].numpy():
    ymin, xmin, ymax, xmax = box
    left = int(xmin * new_img.width)
    right = int(xmax * new_img.width)
    top = int(ymin * new_img.height)
    bottom = int(ymax * new_img.height)
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

new_img.show()

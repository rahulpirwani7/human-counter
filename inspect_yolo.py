from PIL import Image,ImageDraw
import numpy as np
import tensorflow as tf

img = Image.open("test_images/test6.jpeg").convert("RGB")
resized_img = img.resize((640, 640))

input_array = np.array(resized_img, dtype=np.float32) / 255.0
input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...])

print("Input shape:", input_tensor.shape)

model = tf.saved_model.load("yolo26s_saved_model")
print("Model loaded successfully")
predict=model.signatures["serving_default"]
print("Model signature:", predict.structured_outputs)

output = predict(input_tensor)
print("Prediction keys:", output.keys())

print("Prediction shape:",output['output_0'].shape)
predictions = output['output_0'][0].numpy()
print("Predictions shape:", predictions.shape)

candidates=predictions.transpose()
print("Candidates shape:", candidates.shape)

boxes = candidates[:, :4]
scores = candidates[:, 4]
print("Boxes shape:", boxes.shape)
print("Scores shape:", scores.shape)

mask= (scores >= 0.10)

boxes = boxes[mask]
scores = scores[mask]

print("Filtered boxes shape:", boxes.shape)

center_x = boxes[:,0]
center_y = boxes[:,1]
width = boxes[:,2] 
dheight = boxes[:,3]

left_edge = center_x - width / 2    
right_edge = center_x + width / 2
top_edge = center_y - dheight / 2
bottom_edge = center_y + dheight / 2

boxes = np.stack((top_edge, left_edge, bottom_edge, right_edge), axis=1)

print("Final boxes shape:", boxes.shape)
selected_indices= tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=0.5)
print("people count:",len(selected_indices))

final_boxes=tf.gather_nd(boxes, tf.expand_dims(selected_indices, axis=1)).numpy()
print("Final boxes after NMS:", final_boxes)

draw = ImageDraw.Draw(resized_img)
for top, left, bottom, right in final_boxes:
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

resized_img.show()
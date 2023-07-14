import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

weights_url = "https://pjreddie.com/media/files/yolov3.weights"
weights_path = "/Users/mac/Downloads/yolov3.weights"
try:
    urllib.request.urlopen(weights_url)
    print("YOLO weights file already exists.")
except urllib.error.URLError:
    print("Downloading YOLO weights...")
    urllib.request.urlretrieve(weights_url, weights_path)
    print("YOLO weights downloaded successfully.")

config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
config_path = "/Users/mac/Downloads/yolov3.cfg"
try:
    urllib.request.urlopen(config_url)
    print("YOLO configuration file already exists.")
except urllib.error.URLError:
    print("Downloading YOLO configuration...")
    urllib.request.urlretrieve(config_url, config_path)
    print("YOLO configuration downloaded successfully.")

labels_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
labels_path = "/Users/mac/Desktop/coco.names"
try:
    urllib.request.urlopen(labels_url)
    print("COCO class labels file already exists.")
except urllib.error.URLError:
    print("Downloading COCO class labels...")
    urllib.request.urlretrieve(labels_url, labels_path)
    print("COCO class labels downloaded successfully.")

net = cv2.dnn.readNet(weights_path, config_path)


with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

image_path = "/Users/mac/Desktop/Nasir/me.jpg"
image = cv2.imread(image_path)

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)

output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

colors = np.random.uniform(0, 255, size=(len(classes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    confidence = confidences[i]
    color = colors[class_ids[i]]

    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

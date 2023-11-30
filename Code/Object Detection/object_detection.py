import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import time
import torch as th

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
# from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

#import YOLO
from ultralytics import YOLO



# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

model = YOLO(r'C:\Users\ksr20\OneDrive\Desktop\SPEAR\Object_Detection_AI_SPEAR\runs\detect\train\weights\best.pt')

# Run batched inference on a list of images
#change image path to the path of the image you want to test
# results = model([r'C:\Users\ksr20\OneDrive\Desktop\SPEAR\Object_Detection_AI_SPEAR\datasets\coco128\images\train2017\000000000110.jpg'])  # return a list of Results objects
results = YOLO(r'C:\Users\ksr20\OneDrive\Desktop\SPEAR\Object_Detection_AI_SPEAR\runs\detect\predict\bus.jpg')

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image








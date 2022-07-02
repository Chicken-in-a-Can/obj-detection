# For vision
import cv2

# Standard imports
import os
import numpy as np

# TensorFlow imports
import tensorflow as tf
import tensorflow.lite
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor


# TensorFlow imitialization
base_options = core.BaseOptions(file_name="./model.tflite", num_threads=2)
detection_options = processor.DetectionOptions(max_results=1, score_threshold=.5)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

loop = True

# Initialize camera
cam = cv2.VideoCapture("/dev/video0")

while loop: 
    result, image = cam.read()
    
    # Check if camera took picture
    if result:
    
        # Save image to tmp
        cv2.imwrite("/tmp/theimage.png", image)  
    
        # IDK. Machine learning i guess
        image = vision.TensorImage.create_from_file('/tmp/theimage.png')
        detection_result = detector.detect(image)
        
        if str(detection_result) != "DetectionResult(detections=[])":
            # Assign components to variables
            x_orig = detection_result.detections[0].bounding_box.origin_x
            y_orig = detection_result.detections[0].bounding_box.origin_y
            width = detection_result.detections[0].bounding_box.width
            height = detection_result.detections[0].bounding_box.height
            name = detection_result.detections[0].categories[0].category_name

            # Create center location from data
            x_center = x_orig + int(width / 2)
            y_center = y_orig + int(height / 2)
            center = [x_center, y_center, name]

            # Print center
            print(center, end="\r")

        # Remove image from tmp
        os.remove("/tmp/theimage.png")
    else:
        print("Camera failed to take image")
        exit()

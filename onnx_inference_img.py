import numpy as np
from PIL import Image, ImageDraw, ImageColor
import os
import cv2
import time
import onnxruntime as rt
import pandas as pd

image_path = 'test.jpg'
PATH_TO_CKPT = 'model.onnx'
NUM_CLASSES = 1
threshold = 0.5

sess = rt.InferenceSession(PATH_TO_CKPT)



def sess_run(img_data, sess):
    # we want the outputs in this order
        outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
        result = sess.run(outputs, {"input_tensor": img_data})
        num_detections, detection_boxes, detection_scores, detection_classes = result
        return result



def coordinates(width, height, d):
        print('width :' , width)
        print('height :' , height)
        # the box is relative to the image size so we multiply with height and width to get pixels.
        top = d[0] * height
        left = d[1] * width
        bottom = d[2] * height
        right = d[3] * width
        top = int(max(0, np.floor(top + 0.5).astype('int32')))
        left = int(max(0, np.floor(left + 0.5).astype('int32')))
        bottom = int(min(height, np.floor(bottom + 0.5).astype('int32')))
        right = int(min(width, np.floor(right + 0.5).astype('int32')))
        return top, left, bottom, right



def draw_detection(d, c, s, img_arr, height, width):
        """Draw box and label for 1 detection."""
        draw = ImageDraw.Draw(img_arr)
        top, left, bottom, right = coordinates( width,height, d)
        c = str(c)
        c = c[0]
        print(c)
        label = label_map[c]
        s = str(round(s, 2))
        label = label+" : "+s
        print('Class : ', label)
        label_size = draw.textsize(label)
        if top - label_size[1] >= 0:
            text_origin = tuple(np.array([left, top - label_size[1]]))
        else:
            text_origin = tuple(np.array([left, top + 1]))
        color = ImageColor.getrgb("green")
        thickness = 3
        draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
        draw.text(text_origin, label, fill=color)  # , font=font)
        img = np.array(img_arr)
        return img



def img_inference(image_path):
    start = time.time()
    print(image_path)
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    img_arr = Image.fromarray(img)
    image_dir, image_name = os.path.split(image_path)
    #image_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_data = np.expand_dims(img.astype(np.uint8), axis=0)
    result = sess_run(img_data, sess)
    num_detections = result[0]
    detection_boxes = result[1]
    detection_scores = result[2]
    detection_classes = result[3]
    batch_size = num_detections.shape[0]
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            if detection_scores[0][detection] > threshold:
                c = str(detection_classes[batch][detection])
                d = detection_boxes[batch][detection]
                s = detection_scores[0][detection]
                img = draw_detection(d, c, s, img_arr, height, width)
                ress = 'card_output1/'+image_name
                cv2.imwrite(ress, img)
    print("time : ", time.time()-start)
label_map = { '1' : 'class' }   

img_inference(image_path)

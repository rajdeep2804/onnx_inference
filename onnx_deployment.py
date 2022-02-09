import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt
import os
import sys
import cv2
import time
import pathlib
import onnxruntime as rt
import json
import pandas as pd
CWD_PATH = os.getcwd()

class stop_sign_onnx():
    
    
    def __init__(self, onnx_path, stop_and_speed_limit_classes):
        
        self.sess = rt.InferenceSession(onnx_path)
        self.classes = classes
        self.threshold = 0.5
        
    def sess_run(self, img_data, sess):
    # we want the outputs in this order
        outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
        result = self.sess.run(outputs, {"input_tensor": img_data})
        num_detections, detection_boxes, detection_scores, detection_classes = result
        return result
     
    def coordinates(self, draw, d):
        width, height = draw.im.size
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

        
    def json_dump(self, draw, d, c, s, img, frame_number, json_name, stop_and_speed_limit_classes):
        
        top, left, bottom, right = self.coordinates(draw, d)
        label = self.classes[c[0]]
        label1 = label+" : "+str(s)
        print('objects : ', label1)
        print('---\\\\-----detected_sign_frame---\\\\-----',frame_number)
        pixel_location = [left, top, right, bottom]
        entry = {
                  "timestampe" : frame_number,
                    "object" : [{
                            "objName" :label,
                            "confidenceLevel" : float(s),
                            "pixelLocation" : pixel_location,

                        },
                    ]}
        #outfile = open('myfile.json', "a")
        with open(json_name, 'a', encoding='utf-8') as f:
            json.dump(entry, f)

    

    def draw_detection(self, draw, d, c, s, img):
        """Draw box and label for 1 detection."""
        
        top, left, bottom, right = self.coordinates(draw, d)
        label = self.classes[c[0]]
        s = str(round(s, 2))
        label = label+" : "+s
        print('objects : ', label)
        label_size = draw.textsize(label)
        if top - label_size[1] >= 0:
            text_origin = tuple(np.array([left, top - label_size[1]]))
        else:
            text_origin = tuple(np.array([left, top + 1]))
        color = ImageColor.getrgb("green")
        thickness = 1
        draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
        draw.text(text_origin, label, fill=color)  # , font=font)
        img = np.array(img)
        return img

    def _inference_(self, img_array, frame_number, json_name):
        
        img = Image.fromarray(img_array)
        img_data = np.expand_dims(img_array.astype(np.uint8), axis=0)
        result = self.sess_run(img_data, self.sess)
        num_detections = result[0]
        detection_boxes = result[1]
        detection_scores = result[2]
        detection_classes = result[3]
        batch_size = num_detections.shape[0]
        draw = ImageDraw.Draw(img)
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                if detection_scores[0][detection] > self.threshold:
                    c = str(detection_classes[batch][detection])
                    d = detection_boxes[batch][detection]
                    s = detection_scores[0][detection]
                    self.json_dump(draw, d, c, s, img, frame_number, json_name, stop_and_speed_limit_classes)
                    if debug == True :
                        out = self.draw_detection(draw, d, c, s, img)
                        return out

                
    def video_dict(self, cap):
        utc_clock = pd.Timestamp(timestamp_str, tz='utc')
        milliseconds_delta = pd.Timedelta(100, unit='milli')
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret: break
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('--frame_number--',frame_number)
            self._inference_(img, frame_number, json_name)
        cap.release()
        
    def video_debug(self, cap):
        output_path = output_video_dir + '/output_'+video_name
        print('output_path :', output_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret: break
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('--frame_number--',frame_number)
            img = self._inference_(img, frame_number, json_name)
            out.write(img)

        cap.release()
        out.release()
        
        
if __name__ == '__main__':
    
    input_video_path = 'test.mp4'
    output_video_dir = 'output_videos_dir'
    onnx_path = 'model.onnx'
    debug = True
    classes = json.load(open("label_map.txt"))
    
    onnx = stop_sign_onnx(onnx_path, stop_and_speed_limit_classes)
    
    video_path ,video_name = os.path.split(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    total_number_of_frames = int(cv2.VideoCapture.get(cap, property_id))
    print('--total_number_of_frames--', total_number_of_frames)
    json_name = video_name[:-4]+'.json'
    print('----------json------', json_name)
        
    if debug == False : 
        onnx.video_dict(cap)
    else:
        onnx.video_debug(cap) 
       
      
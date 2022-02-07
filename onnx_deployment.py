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
CWD_PATH = os.getcwd()

class stop_sign_onnx():
    
    
    def __init__(self, onnx_path, stop_and_speed_limit_classes):
        
        self.sess = rt.InferenceSession(onnx_path)
        self.stop_and_speed_limit_classes = stop_and_speed_limit_classes
        
        
    def sess_run(self, img_data, sess): # onnx inference
    # we want the outputs in this order
        outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
        result = self.sess.run(outputs, {"input_tensor": img_data})
        num_detections, detection_boxes, detection_scores, detection_classes = result
        return result

    def creating_json_file(self, tail): # Creating json file with same video name
        json_name = tail[:-4]
        json_name = json_name+'.json'
        return json_name


    def json_dump(self, draw, d, c, s, img, frame_number, json_name, stop_and_speed_limit_classes): # dumping json data 
        """Draw box and label for 1 detection."""
        width, height = draw.im.size
        print('width :' , width)
        print('height :' , height)
        top = int(d[0] * height)
        left = int(d[1] * width)
        bottom = int(d[2] * height)
        right = int(d[3] * width)
        s = (s * 100 + 0.5) / 100.0
        label = self.stop_and_speed_limit_classes[c]
        print('---\\\\-----detected_sign_frame---\\\\-----',frame_number)
        pixel_location = [left, top, right, bottom]
        
        entry = {
                  "timestampe" : frame_number*100,
                    "object" : {
                            "objName" :label,
                            "confidenceLevel" : s,
                            "pixelLocation" : pixel_location,

                        },
                     }
        with open(json_name, 'a', encoding='utf-8') as f:
            json.dump(entry, f)


    def draw_detection(self, draw, d, c, s, img): # draw detected bounding box
        """Draw box and label for 1 detection."""
        width, height = draw.im.size
        print('width :' , width)
        print('height :' , height)
        top = d[0] * height
        left = d[1] * width
        bottom = d[2] * height
        right = d[3] * width
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
        right = min(width, np.floor(right + 0.5).astype('int32'))
        label = self.stop_and_speed_limit_classes[c]
        s = str(round(s, 2))
        label1 = label+" : "+s
        print('objects : ', label1)
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

    def save_inference_video(self, image_path, sess, frame_number, json_name): # Saving inference data
        start_time = time.time()
        img = Image.fromarray(image_path)
        img_data = image_path
        img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
        result = self.sess_run(img_data, sess)
        num_detections = result[0]
        detection_boxes = result[1]
        detection_scores = result[2]
        detection_classes = result[3]
        batch_size = num_detections.shape[0]
        #print(batch_size)
        draw = ImageDraw.Draw(img)
        #print(draw)
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                if detection_scores[0][detection] > 0.5:
                    c = detection_classes[batch][detection]
                    d = detection_boxes[batch][detection]
                    s = detection_scores[0][detection]
                    self.json_dump(draw, d, c, s, img, frame_number, json_name, stop_and_speed_limit_classes)
                    out = self.draw_detection(draw, d, c, s, img)
                    return out
        end_time = time.time() - start_time
        print('onnx_inference_time : ',end_time )

    def save_video(self,TEST_VIDEO_PATHS): # loading video and saving video 
        start = time.time()
        for video_path in TEST_VIDEO_PATHS:
            video_path = str(video_path)
            head_tail = os.path.split(video_path)
            tail = head_tail[1]
            cap = cv2.VideoCapture(video_path)
            output_path = output_video_dir + '/output_'+tail
            print('output_path :', output_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)
            total_number_of_frames = int(cv2.VideoCapture.get(cap, property_id))
            print('--total_number_of_frames--', total_number_of_frames)

            json_name = self.creating_json_file(tail)
            print('----------json------', json_name)
            while (cap.isOpened()):
                ret, img = cap.read()
                if not ret: break
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print('--frame_number--',frame_number)
                img = self.save_inference_video(img, self.sess, frame_number, json_name)
                out.write(img)
            cap.release()
            out.release()   
            end = time.time() - start
        print('total_time : ', end, 'sec')   

               
        
if __name__ == '__main__':
    
    input_video_path = 'input_video_path'
    output_video_dir = 'output_videos_path'
    onnx_path = 'model.onnx'
    classes = {
                1: 'class_1',
                2: 'class_2',
                3: 'class_3',
                4: 'class_4',
                5: 'class_5',
                6: 'class_6',
               }
    
    PATH_TO_TEST_VIDEO_DIR = pathlib.Path(input_video_path)
    TEST_VIDEO_PATHS = sorted(list(PATH_TO_TEST_VIDEO_DIR.glob("*.mp4")))
    onnx = stop_sign_onnx(onnx_path, stop_and_speed_limit_classes)
    onnx.save_video(TEST_VIDEO_PATHS)
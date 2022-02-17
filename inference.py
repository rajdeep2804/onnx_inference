from onnx_deployment import Stop_sign_onnx
import json
input_video_path = 'video.mp4'
output_video_dir = 'output_dir'
onnx_path = 'model.onnx'
timestamp_str = '2022-01-25T11:04:00.000Z'
label_map_path = "label_map.txt"
debug = True


stop_and_speed_limit_classes = json.load(open(label_map_path))
    
onnx = Stop_sign_onnx(onnx_path, stop_and_speed_limit_classes)

onnx.video_processing(input_video_path, debug, timestamp_str, output_video_dir)

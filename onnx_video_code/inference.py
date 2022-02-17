from onnx_deployment import Stop_sign_onnx
import json
input_video_path = 'video.mp4'
output_video_dir = 'output_dir'
onnx_path = 'model.onnx'
label_map_path = "label_map.txt"
debug = True


classes = json.load(open(label_map_path))
    
onnx = Stop_sign_onnx(onnx_path, classes)

onnx.video_processing(input_video_path, debug, output_video_dir)

# onnx_inference

## Description

### ONNX

ONNX (The Open Neural Network Exchange Format) is an intermediary machine learning framework used to convert between different machine learning frameworks. ONNX is a powerful and open standard for preventing framework lock-in and ensuring that you the models you develop will be usable in the long run.
The end result of a trained deep learning algorithm is a model file that efficiently represents the relationship between input data and output predictions. A neural network is one of the most powerful ways to generate these predictive models but can be difficult to build in to production systems. Most often, these models exist in a data format such as .pth file or an HD5 file. Oftentimes you want these models to be portable so that you can deploy them in environments that might be different than where you initially trained the model.

### ONNX Overview

At a high level, ONNX is designed to allow framework interoporability. There are many excellent machine learning libraries in various languages — PyTorch, TensorFlow, MXNet, and Caffe are just a few that have become very popular in recent years, but there are many others as well.
The idea is that you can train a model with one tool stack and then deploy it using another for inference and prediction. To ensure this interoperability you must export your model in the model.onnx format which is serialized representation of the model in a protobuf file. Currently there is native support in ONNX for PyTorch, CNTK, MXNet, and Caffe2 but there are also converters for TensorFlow and CoreML.

### ONNX Runtime

ONNX Runtime is a high-performance inference engine for machine learning models in the ONNX format on Linux, Windows, and Mac. ONNX is an open format for deep learning and traditional machine learning models that Microsoft co-developed with Facebook and AWS.

## Requirements

- Python 3.7.6
- onnx 1.10.2
- onnxruntime 1.6.0
- tf2onnx 1.9.3
- tensorflow-gpu 2.4.1


Further you can check your device compatibility and versions from [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

### Converting tf2 exported model weights to onnx model

``` python -m tf2onnx.convert --saved-model exported_model_path/saved_model --opset 13 --fold_const --output model.onnx ```

### Inference on image data or video data

you can use "onnx_img_video_inference.ipynb" which will help you under required input and output tensor shape to get inference results on your test data.
you can also refere to this [jupyter notebook ](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb)

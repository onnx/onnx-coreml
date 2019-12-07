### Converting PyTorch model to CoreML model (Short version)
Converting PyTorch model into CoreML model is a two step process
1. Convert PyTorch model into ONNX model
  - PyTorch model can be converted into ONNX model using `torch.onnx.export`
  - Reference: https://pytorch.org/docs/stable/onnx.html#id2
  - Tools required: [PyTorch](https://pytorch.org/get-started/locally/)
2. Converting ONNX model into CoreML model
  - We will be using model converted in step 1 using `onnx_coreml.convert`
  - Tools required: [onnx-coreml](https://pypi.org/project/onnx-coreml/)
  

### Converting PyTorch model to CoreML model (Long version)
1. PyTorch to ONNX conversion
  - In this example, we are creating a small test model
    ```
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      # Step 0 - (a) Define ML Model
      class small_model(nn.Module):
          def __init__(self):
              super(small_model, self).__init__()
              self.fc1 = nn.Linear(768, 256)
              self.fc2 = nn.Linear(256, 10)

          def forward(self, x):
              y = F.relu(self.fc1(x))
              y = F.softmax(self.fc2(y))
              return y
    ```
  - Load model
    ```
      # Step 0 - (b) Create model or Load from distk
      model = small_model()
      dummy_input = torch.randn(768)
    ```
  - Convert From PyTorch to ONNX
    ```
      # Step 1 - PyTorch to ONNX model
      torch.onnx.export(model, dummy_input, './small_model.onnx')
    ```
2. ONNX to CoreML
    ```
      # Step 2 - ONNX to CoreML model
      mlmodel = convert(model='./small_model.onnx', minimum_ios_deployment_target='13')
      # Save converted CoreML model
      mlmodel.save('small_model.mlmodel')
    ```

### What about frameworks other than PyTorch?
  - Step 1 can be replaced by respective framework to ONNX converter,
  - Once, you have a ONNX model, you can use `onnx_coreml.convert` to get CoreML model
    which can be deployed on device.
    
### More examples demonstrating prediction and validating converted model's correctness
  - BERT: https://github.com/onnx/onnx-coreml/blob/master/examples/BERT.ipynb
  - All: https://github.com/onnx/onnx-coreml/blob/master/examples/

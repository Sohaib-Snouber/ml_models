import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model = "color_classifier.onnx"
onnx.checker.check_model(onnx_model)  # Validate ONNX format

# Force TensorRT as the first execution provider
execution_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

# Create an ONNX Runtime session with TensorRT as the priority
session = ort.InferenceSession(onnx_model, providers=execution_providers)

# Prepare dummy input (matching the model input shape)
dummy_input = np.random.rand(1, 3).astype(np.float32)  # 1 sample with 3 features (RGB)

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input})

print("ONNX model inference successful! Output:", output)

# Print Execution Providers
print("Available Execution Providers:", ort.get_available_providers())
print("ONNX is running on:", session.get_providers())

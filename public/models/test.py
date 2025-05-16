import onnxruntime
import numpy as np

# Load the ONNX model
model_path = "./yolo11n-2.onnx" # Make sure this path is correct
session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider']) # Or ['CUDAExecutionProvider'] if you have GPU

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape # e.g., [1, 3, 640, 640] (dynamic might show -1 or 'batch_size', etc.)
print(f"Input Name: {input_name}")
print(f"Expected Input Shape: {input_shape}")

# Create a dummy input tensor (adjust shape and type as per your model)
# Assuming input is [1, 3, 640, 640] and float32
# For dynamic axes, you might need to be more specific or use a fixed size for testing.
# Let's use a fixed 640x640 for this test if your input_shape shows dynamic height/width.
batch_size = 1
channels = 3
height = 640
width = 640
dummy_input = np.random.rand(batch_size, channels, height, width).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})

# 'outputs' is a list of numpy arrays. For YOLO, it's often just one output tensor.
output_tensor = outputs[0]
print(f"Output Tensor Shape: {output_tensor.shape}") # Should be [1, 84, 8400] or similar

# Inspect the output data, particularly scores
# Output format [batch_size, num_features, num_candidates]
# num_features = 4 (bbox) + 1 (obj_score) + num_classes
# Objectness score is at index 4 of the num_features dimension.
# Class scores start at index 5.

num_candidates_to_check = 10
objectness_scores = output_tensor[0, 4, :num_candidates_to_check]
print(f"First {num_candidates_to_check} Objectness Scores: \n{objectness_scores}")

class_scores_candidate0 = output_tensor[0, 5:, 0] # Class scores for the first candidate
print(f"Class Scores for Candidate 0 (first 10): \n{class_scores_candidate0[:10]}")

if np.all(objectness_scores == 0):
    print("\nWARNING: All sampled objectness scores are zero in Python ONNX Runtime!")
else:
    print("\nSUCCESS: Non-zero objectness scores found in Python ONNX Runtime.")

import tensorflow as tf

# Load and inspect the exported model
model = tf.saved_model.load("exported_model")
print("Available signatures:", list(model.signatures.keys()))

# Get the default serving signature
serving_fn = model.signatures["serving_default"]
print("\nInput signature:")
for key, spec in serving_fn.structured_input_signature[1].items():
    print(f"  {key}: {spec}")

print("\nOutput signature:")
for key, spec in serving_fn.structured_outputs.items():
    print(f"  {key}: {spec}")

# Test with dummy input to see output shape
import numpy as np
dummy_input = tf.constant(np.zeros((1, 19, 19, 2), dtype=np.float32))
output = serving_fn(dummy_input)
print(f"\nOutput keys: {list(output.keys())}")
for key, tensor in output.items():
    print(f"  {key}: shape={tensor.shape}, sample values={tensor.numpy().flatten()[:5]}")
import os
import tensorflow as tf
from train import create_model, MODEL_FILE # Import from your training script

# --- Configuration ---
EXPORT_DIRECTORY = 'exported_model'

def main():
    """Loads the trained weights and saves the model in SavedModel format."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model weights file not found at '{MODEL_FILE}'")
        print("Please train the model first using the training script.")
        return

    print("Creating model architecture...")
    # Use the same parameters as your trained model
    model = create_model(num_residual_blocks=5, filters=128)

    print(f"Loading weights from '{MODEL_FILE}'...")
    model.load_weights(MODEL_FILE)

    print(f"Exporting model to '{EXPORT_DIRECTORY}'...")
    # Save the entire model to the specified directory
    model.export(EXPORT_DIRECTORY)

    print("\nExport complete.")
    print(f"Model saved in SavedModel format at: {os.path.abspath(EXPORT_DIRECTORY)}")

if __name__ == '__main__':
    main()


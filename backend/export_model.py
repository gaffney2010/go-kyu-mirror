import os
import tensorflow as tf
import shutil

# --- Configuration ---
# The final directory for the deployable model (for the Java app)
EXPORT_DIRECTORY = 'exported_model_anyk'

# The .keras file where train.py saves its latest checkpoint
CHECKPOINT_FILE = 'training_checkpoint.keras'

def main():
    """
    Loads the complete model from a .keras checkpoint file and saves it
    in the SavedModel directory format required by the Java application.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_FILE}'")
        print("Please ensure training has run and saved at least one checkpoint.")
        return

    print(f"Loading complete model from checkpoint: '{CHECKPOINT_FILE}'")
    
    # Load the entire model from the .keras file.
    model = tf.keras.models.load_model(CHECKPOINT_FILE)

    print("Model loaded successfully.")
    
    # If the export directory already exists, remove it for a clean save.
    if os.path.exists(EXPORT_DIRECTORY):
        print(f"Removing existing export directory: '{EXPORT_DIRECTORY}'")
        shutil.rmtree(EXPORT_DIRECTORY)
    
    print(f"Saving final deployable model to '{EXPORT_DIRECTORY}'...")
    
    # --- FIX ---
    # Use model.export() to save in the SavedModel directory format.
    # The error message explicitly recommends this for TFServing/TFLite/etc.
    model.export(EXPORT_DIRECTORY)

    print("\n✅ Export complete.")
    print(f"Deployable model is ready at: {os.path.abspath(EXPORT_DIRECTORY)}")

if __name__ == '__main__':
    main()


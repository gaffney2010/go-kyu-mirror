import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.models import Model

# --- GPU/CPU Configuration ---
# By default, TensorFlow will try to use a GPU if it's available.
# If you want to force CPU execution for training (not recommended for performance),
# you can uncomment the following line:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Configuration ---
# Point this to the directory containing the preprocessed data for the desired rank.
TFRECORD_DIRECTORY = 'data/tfrecords/10k'
MODEL_FILE = 'go_bot_10k_predictor.h5'  # Where to save the trained model.

BOARD_SIZE = 19
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 2)  # Board state: player's stones, opponent's stones
NUM_OUTPUTS = BOARD_SIZE * BOARD_SIZE + 1  # 361 board positions + 1 for pass

# Training parameters
BATCH_SIZE = 128
EPOCHS = 10 # You can train for more epochs since reading data is now very fast.
BUFFER_SIZE = 10000 # For shuffling the dataset. A larger buffer means better shuffling.

def parse_tfrecord_fn(example):
    """Parses a single record from a TFRecord file."""
    feature_description = {
        'board_state': tf.io.FixedLenFeature([], tf.string),
        'next_move': tf.io.FixedLenFeature([], tf.int64),
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    board_state = tf.io.parse_tensor(example['board_state'], out_type=tf.float32)
    board_state = tf.reshape(board_state, INPUT_SHAPE)
    
    next_move = tf.cast(example['next_move'], tf.int32)
    next_move_one_hot = tf.one_hot(next_move, depth=NUM_OUTPUTS)
    
    return board_state, next_move_one_hot

def create_dataset(tfrecord_dir):
    """Creates a tf.data.Dataset from a directory of TFRecord files."""
    file_paths = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord*"))
    if not file_paths:
        raise ValueError(f"No TFRecord files found in directory: {tfrecord_dir}")
        
    print(f"Found {len(file_paths)} TFRecord files.")
    
    # Create a dataset from the files
    dataset = tf.data.TFRecordDataset(file_paths, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    
    # Shuffle, parse, batch, and prefetch for performance
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def residual_block(x, filters=128):
    """A single residual block for the ResNet architecture."""
    res = x
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = ReLU()(x)
    return x

def create_model(num_residual_blocks=5, filters=128):
    """Creates a ResNet model similar to modern Go engines."""
    inputs = Input(shape=INPUT_SHAPE)
    
    # Stem: Initial convolution
    x = Conv2D(filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Trunk: A stack of residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, filters)
        
    # Policy Head
    policy = Conv2D(2, 1, padding='same')(x)
    policy = BatchNormalization()(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    policy = Dense(NUM_OUTPUTS, activation='softmax', name='policy')(policy)
    
    model = Model(inputs=inputs, outputs=policy)
    return model

def main():
    """Main function to load data, create a model, and start training."""
    print("Creating dataset from TFRecords...")
    try:
        train_dataset = create_dataset(TFRECORD_DIRECTORY)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the preprocessing script and that the TFRECORD_DIRECTORY is correct.")
        return

    print("Creating model...")
    model = create_model()
    
    # Check if a model file already exists to continue training
    if os.path.exists(MODEL_FILE):
        print(f"Found existing model at '{MODEL_FILE}'. Loading weights to continue training.")
        model.load_weights(MODEL_FILE)
        
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()

    # Callback to save the model after every epoch
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_FILE,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    print("\nStarting training...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback]
    )
    print("\nTraining complete.")
    print(f"Final model saved to '{MODEL_FILE}'")

if __name__ == '__main__':
    main()

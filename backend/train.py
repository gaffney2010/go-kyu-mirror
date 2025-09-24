import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.models import Model

# --- Configuration ---
TFRECORD_BASE_DIR = 'data/tfrecords'
TFRECORD_FILE = 'data.tfrecord-00000'
RANKS = [f"{i}k" for i in range(1, 16)]

# The LATEST training checkpoint will be saved as a single, reliable .keras file
CHECKPOINT_FILE = 'training_checkpoint.keras'

# Model and training parameters
BOARD_SIZE = 19
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 2)
NUM_OUTPUTS = BOARD_SIZE * BOARD_SIZE + 1
BATCH_SIZE = 128
EPOCHS = 10
BUFFER_SIZE = 10000
SAVE_FREQ_BATCHES = 1200 # Frequency for saving checkpoints

# --- Model Definition with Explicit Naming ---
def residual_block(x, filters=128, block_num=0):
    res = x
    x = Conv2D(filters, 3, padding='same', name=f'res{block_num}_conv1')(x)
    x = BatchNormalization(name=f'res{block_num}_bn1')(x)
    x = ReLU(name=f'res{block_num}_relu1')(x)
    x = Conv2D(filters, 3, padding='same', name=f'res{block_num}_conv2')(x)
    x = BatchNormalization(name=f'res{block_num}_bn2')(x)
    x = Add(name=f'res{block_num}_add')([x, res])
    x = ReLU(name=f'res{block_num}_relu2')(x)
    return x

def create_multi_head_model(num_ranks, num_residual_blocks=5, filters=128):
    inputs = Input(shape=INPUT_SHAPE, name='input_1')
    x = Conv2D(filters, 3, padding='same', name='stem_conv')(inputs)
    x = BatchNormalization(name='stem_bn')(x)
    x = ReLU(name='stem_relu')(x)
    for i in range(num_residual_blocks):
        x = residual_block(x, filters, block_num=i)
    policy_heads = {}
    for i in range(num_ranks):
        layer_name = f'policy_rank_{i}'
        policy = Conv2D(2, 1, padding='same', name=f'policy_conv_{i}')(x)
        policy = BatchNormalization(name=f'policy_bn_{i}')(policy)
        policy = ReLU(name=f'policy_relu_{i}')(policy)
        policy = Flatten(name=f'policy_flatten_{i}')(policy)
        policy = Dense(NUM_OUTPUTS, activation='softmax', name=layer_name)(policy)
        policy_heads[layer_name] = policy
    model = Model(inputs=inputs, outputs=policy_heads)
    return model

# --- Data Loading Functions ---
def parse_tfrecord_fn(example):
    feature_description = {
        'board_state': tf.io.FixedLenFeature([], tf.string),
        'next_move': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    board_state = tf.io.decode_raw(example['board_state'], out_type=tf.float32)
    board_state = tf.reshape(board_state, INPUT_SHAPE)
    next_move = tf.cast(example['next_move'], tf.int32)
    next_move_one_hot = tf.one_hot(next_move, depth=NUM_OUTPUTS)
    return board_state, next_move_one_hot

def create_rank_dataset(rank):
    file_path = os.path.join(TFRECORD_BASE_DIR, rank, TFRECORD_FILE)
    if not os.path.exists(file_path):
        return None
    dataset = tf.data.TFRecordDataset(file_path, num_parallel_reads=tf.data.AUTOTUNE)
    return dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

def create_combined_dataset():
    """
    Creates a combined dataset by interleaving samples from all ranks
    to ensure perfectly balanced shuffling from the start.
    """
    all_datasets, available_ranks = [], []
    for i, rank in enumerate(RANKS):
        dataset = create_rank_dataset(rank)
        if dataset is not None:
            # Add the rank index to each sample within its own dataset
            rank_dataset = dataset.map(
                lambda board, move: (board, move, tf.constant(i, dtype=tf.int32)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            all_datasets.append(rank_dataset)
            available_ranks.append(rank)
            
    if not all_datasets:
        raise ValueError("No valid TFRecord files found for any rank!")
        
    print(f"Found data for ranks: {available_ranks}")
    print("Interleaving datasets for balanced shuffling...")

    # Use sample_from_datasets to create a perfectly mixed stream of data
    # This pulls from all rank datasets simultaneously.
    combined_dataset = tf.data.Dataset.sample_from_datasets(
        all_datasets,
        stop_on_empty_dataset=True # Stop when the smallest dataset is exhausted
    )
    
    # Now, the shuffle buffer will be filled with a mix of all ranks
    combined_dataset = combined_dataset.shuffle(BUFFER_SIZE)
    combined_dataset = combined_dataset.batch(BATCH_SIZE)
    combined_dataset = combined_dataset.prefetch(tf.data.AUTOTUNE)
    
    return combined_dataset, available_ranks

# --- Main Execution ---
def main():
    """Main function to load data, create model, and train with reliable, frequent checkpointing."""
    train_dataset, available_ranks = create_combined_dataset()
    num_available_ranks = len(available_ranks)
    
    # Check if a checkpoint FILE exists to resume training
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming training from checkpoint: {CHECKPOINT_FILE}")
        model = tf.keras.models.load_model(CHECKPOINT_FILE)
    else:
        print("Starting new training session...")
        model = create_multi_head_model(num_available_ranks)

    losses = {f'policy_rank_{i}': 'categorical_crossentropy' for i in range(num_available_ranks)}
    metrics = {f'policy_rank_{i}': ['accuracy'] for i in range(num_available_ranks)}
    model.compile(optimizer='adam', loss=losses, metrics=metrics)
    model.summary()

    # This callback saves the ENTIRE MODEL to a single .keras file frequently.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILE,      # Saves to the correct .keras file
        save_weights_only=False,       # Saves the full model, which is reliable
        save_freq=SAVE_FREQ_BATCHES    # Saves every N batches
    )

    print(f"\nTraining... A checkpoint will be saved to '{CHECKPOINT_FILE}' every {SAVE_FREQ_BATCHES} batches.")
    
    def transform_for_multihead_and_weighting(board_states, moves, rank_indices):
        outputs = {f'policy_rank_{i}': moves for i in range(num_available_ranks)}
        sample_weights = {f'policy_rank_{i}': tf.cast(tf.equal(rank_indices, i), tf.float32) for i in range(num_available_ranks)}
        return board_states, outputs, sample_weights
    
    weighted_dataset = train_dataset.map(transform_for_multihead_and_weighting, num_parallel_calls=tf.data.AUTOTUNE)

    model.fit(
        weighted_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback]
    )
    print(f"\nTraining complete. Latest model checkpoint saved at: {os.path.abspath(CHECKPOINT_FILE)}")

if __name__ == '__main__':
    main()


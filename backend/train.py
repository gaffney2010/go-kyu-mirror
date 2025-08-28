import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization, ReLU, Add

# --- Configuration ---
SGF_DIRECTORY = 'data/sgfs-by-date'  # Directory containing your SGF files
TARGET_RANK = '10k'
BOARD_SIZE = 19
# The input shape now has 2 channels: one for the current player's stones, and one for the opponent's.
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 2)
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE + 1  # 19x19 board + 1 for pass
EPOCHS = 1 # Set to 1 for single-pass training
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000 # Number of elements from which the new dataset will sample.

# --- New Configuration for Controlling Training Runs ---
# Set to an integer to use a subset of files, or None to use all files.
MAX_FILES_TO_USE = 10000
# Set to True to load weights from a previously saved model and continue training.
CONTINUE_TRAINING = True
MODEL_FILE = 'go_next_move_predictor.h5'
# A sister file to store the training state, like the file offset.
STATE_FILE = 'training_state.json'


def get_player_rank(sgf_game, player_color):
    """Extracts the rank of a player from the SGF file."""
    root_node = sgf_game.get_root()
    rank_property = f'BR' if player_color == 'b' else f'WR'
    if root_node.has_property(rank_property):
        rank_value = root_node.get(rank_property)
        if isinstance(rank_value, bytes):
            # If the value is bytes, decode it to a string.
            return rank_value.decode('utf-8', errors='ignore')
        # If it's already a string, return it directly.
        return str(rank_value)
    return None

def process_sgf(file_path):
    """
    Processes a single SGF file.
    Extracts game data if one of the players has the target rank.
    If only one player has the target rank, only their moves are used.
    Returns a list of (board_state, next_move) tuples.
    This function is called by the generator.
    """
    try:
        with open(file_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:
        # Silently skip corrupted or unreadable files
        return []

    if game.get_size() != BOARD_SIZE:
        return []

    b_rank = get_player_rank(game, 'b')
    w_rank = get_player_rank(game, 'w')

    b_is_target = b_rank and TARGET_RANK in b_rank
    w_is_target = w_rank and TARGET_RANK in w_rank

    # If neither player is the target rank, skip the game.
    if not (b_is_target or w_is_target):
        return []

    try:
        board, plays = sgf_moves.get_setup_and_moves(game)
    except Exception as e:
        # Skip games with move processing errors
        return []

    game_data = []
    for i, (color, move) in enumerate(plays):
        # Determine if we should record this move based on player rank
        record_move = False
        if b_is_target and w_is_target:
            # If both are target rank, record all moves
            record_move = True
        elif b_is_target and color == 'b':
            # If only black is target, only record black's moves
            record_move = True
        elif w_is_target and color == 'w':
            # If only white is target, only record white's moves
            record_move = True

        if record_move:
            if move is None:  # Pass move
                next_move_index = BOARD_SIZE * BOARD_SIZE
            else:
                row, col = move
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    # This move is invalid, so we won't record it.
                    record_move = False
                else:
                    next_move_index = row * BOARD_SIZE + col

            if record_move:
                # Create a board state representation (2 channels: player's stones, opponent's stones)
                board_state = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
                opponent_color = 'w' if color == 'b' else 'b'
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        stone = board.get(r, c)
                        if stone == color:
                            board_state[r, c, 0] = 1.0  # Player's stones
                        elif stone == opponent_color:
                            board_state[r, c, 1] = 1.0  # Opponent's stones
                game_data.append((board_state, np.int32(next_move_index)))

        # Always apply the move to the board for the next iteration, regardless of whether we recorded it.
        try:
            if move:
                board.play(move[0], move[1], color)
        except Exception as e:
            # This can happen with illegal moves in SGFs, we'll just stop processing here
            break

    return game_data

def data_generator(file_paths):
    """
    A generator function that yields training examples from SGF files.
    """
    for file_path in file_paths:
        for board_state, next_move in process_sgf(file_path):
            yield board_state, next_move

def residual_block(x, filters):
    """A residual block used in ResNet architectures."""
    # Main path
    shortcut = x
    # Pre-activation style
    y = BatchNormalization()(x)
    y = ReLU()(y)
    y = Conv2D(filters, 3, padding='same')(y)
    
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(filters, 3, padding='same')(y)
    
    # Add the shortcut (input) to the output of the convolutions
    y = Add()([shortcut, y])
    return y

def create_model(num_res_blocks=5, num_filters=64):
    """
    Creates the CNN model with a ResNet architecture similar to KataGo.
    """
    inputs = Input(shape=INPUT_SHAPE)
    
    # --- Stem ---
    # An initial convolutional block to process the input board state.
    x = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # --- Trunk ---
    # A stack of residual blocks to form the deep body of the network.
    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)
        
    # --- Policy Head ---
    # Takes the features from the trunk and outputs move probabilities.
    policy_head = Conv2D(2, (1, 1), padding='same')(x)
    policy_head = BatchNormalization()(policy_head)
    policy_head = ReLU()(policy_head)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(NUM_CLASSES, activation='softmax', name='policy_output')(policy_head)
    
    # --- Create and Compile Model ---
    model = Model(inputs=inputs, outputs=policy_head)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    """Main function to load data, create and train the model."""
    print("Finding SGF files...")
    all_sgf_files = []
    for root, _, files in os.walk(SGF_DIRECTORY):
        for file in files:
            if file.endswith('.sgf'):
                all_sgf_files.append(os.path.join(root, file))

    if not all_sgf_files:
        print(f"No SGF files found in '{SGF_DIRECTORY}'.")
        return

    print(f"Found {len(all_sgf_files)} total SGF files.")
    # We shuffle the list of all files to ensure that when we take a subset,
    # it's a random sample. This should only be done once for a dataset.
    # For true resumable training, this list should be static.
    random.shuffle(all_sgf_files)

    file_offset = 0
    # If continuing, load the previous state.
    if CONTINUE_TRAINING and os.path.exists(STATE_FILE):
        print(f"Found state file '{STATE_FILE}'. Loading previous state.")
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            file_offset = state.get('file_offset', 0)
        print(f"Resuming from file offset: {file_offset}")

    # Apply the offset to get the list of files for this run.
    remaining_files = all_sgf_files[file_offset:]

    # Determine how many files to use in this specific training session.
    if MAX_FILES_TO_USE is not None and MAX_FILES_TO_USE < len(remaining_files):
        train_files = remaining_files[:MAX_FILES_TO_USE]
        print(f"Using a subset of {len(train_files)} files for this run.")
    else:
        train_files = remaining_files
        print(f"Using all {len(train_files)} remaining files.")

    if not train_files:
        print("No more files to process. Training is complete.")
        return

    print(f"This session will use {len(train_files)} files for training.")

    # Create a mapping function to one-hot encode the labels
    def preprocess(board, label):
        return board, tf.one_hot(label, depth=NUM_CLASSES)

    # --- Create Training Dataset ---
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_files),
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Creating model...")
    model = create_model()

    # If continuing training, load the existing model weights.
    if CONTINUE_TRAINING and os.path.exists(MODEL_FILE):
        print(f"Loading weights from {MODEL_FILE} to continue training.")
        model.load_weights(MODEL_FILE)

    model.summary()

    print("Starting training...")
    model.fit(train_dataset,
              epochs=EPOCHS)

    print("Training complete for this session.")
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Update and save the new offset for the next run.
    new_offset = file_offset + len(train_files)
    with open(STATE_FILE, 'w') as f:
        json.dump({'file_offset': new_offset}, f)
    print(f"Updated state file. Next run will start from file offset: {new_offset}")


if __name__ == '__main__':
    main()

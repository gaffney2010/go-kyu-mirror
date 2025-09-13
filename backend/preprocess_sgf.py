import os
# --- GPU Configuration ---
# Set this environment variable BEFORE importing TensorFlow to disable the GPU.
# This is the most reliable way to prevent CUDA initialization errors on CPU-only machines
# or machines with driver issues.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import numpy as np
import tensorflow as tf
from sgfmill import sgf, sgf_moves
import logging

# --- Logging Configuration ---
# Set up logging to write to a file instead of the console.
logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("CUDA_VISIBLE_DEVICES set to -1. Forcing CPU execution.")


# --- Configuration ---
SGF_DIRECTORY = 'data/sgfs-by-date'
TFRECORD_OUTPUT_DIR = 'data/tfrecords'
# Process all ranks from 1k to 15k.
TARGET_RANKS = [f"{i}k" for i in range(1, 16)]
BOARD_SIZE = 19
NUM_SHARDS = 100  # Split the data into 100 separate files for each rank.
# A state file to track which SGFs have already been processed.
STATE_FILE = 'preprocessing_state.txt'

def get_player_rank(sgf_game, player_color):
    """Extracts the rank of a player from the SGF file."""
    root_node = sgf_game.get_root()
    rank_property = f'BR' if player_color == 'b' else f'WR'
    if root_node.has_property(rank_property):
        rank_value = root_node.get(rank_property)
        if isinstance(rank_value, bytes):
            return rank_value.decode('utf-8', errors='ignore')
        return str(rank_value)
    return None

def process_sgf(file_path, target_ranks_set):
    """
    Processes a single SGF file and yields (rank, board_state, next_move) tuples.
    """
    try:
        with open(file_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception:
        return

    if game.get_size() != BOARD_SIZE:
        return

    b_rank = get_player_rank(game, 'b')
    w_rank = get_player_rank(game, 'w')

    b_is_target = b_rank in target_ranks_set
    w_is_target = w_rank in target_ranks_set

    if not (b_is_target or w_is_target):
        return

    try:
        board, plays = sgf_moves.get_setup_and_moves(game)
    except Exception:
        return

    for color, move in plays:
        rank_to_record = None
        # Determine if the current player's move should be recorded.
        if color == 'b' and b_is_target:
            rank_to_record = b_rank
        elif color == 'w' and w_is_target:
            rank_to_record = w_rank

        if rank_to_record:
            if move is None:
                next_move_index = BOARD_SIZE * BOARD_SIZE
            else:
                row, col = move
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    continue
                next_move_index = row * BOARD_SIZE + col

            board_state = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
            opponent_color = 'w' if color == 'b' else 'b'
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    stone = board.get(r, c)
                    if stone == color:
                        board_state[r, c, 0] = 1.0
                    elif stone == opponent_color:
                        board_state[r, c, 1] = 1.0
            
            yield rank_to_record, board_state, next_move_index

        try:
            if move:
                board.play(move[0], move[1], color)
        except Exception:
            break

# Helper functions for serializing data to TFRecord format
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(board_state, next_move):
    """Creates a tf.train.Example message ready to be written to a file."""
    feature = {
        'board_state': _bytes_feature(tf.io.serialize_tensor(board_state)),
        'next_move': _int64_feature(next_move),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_to_tfrecord_append(shard_path, moves):
    """
    Safely append moves to a TFRecord file.
    Since TFRecordWriter doesn't support direct append mode, we collect all existing
    data and rewrite the entire file with the new data appended.
    """
    try:
        # Read existing records if the file exists
        existing_examples = []
        if tf.io.gfile.exists(shard_path):
            try:
                for record in tf.data.TFRecordDataset(shard_path, compression_type="GZIP"):
                    existing_examples.append(record.numpy())
            except Exception as e:
                # If we can't read the existing file, log a warning but continue
                logging.warning(f"Could not read existing data from {shard_path}: {e}")
                existing_examples = []
        
        # Serialize new moves
        new_examples = []
        for board_state, next_move in moves:
            example = serialize_example(board_state, next_move)
            new_examples.append(example)
        
        # Write all examples (existing + new) to the file
        with tf.io.TFRecordWriter(shard_path, options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
            # Write existing examples first
            for example in existing_examples:
                writer.write(example)
            # Write new examples
            for example in new_examples:
                writer.write(example)
                
    except Exception as e:
        logging.error(f"Failed to write to {shard_path}: {e}")
        raise

def main():
    """Main function to find, process, and save SGF data to TFRecords."""
    # --- Create Directories for each Rank ---
    for rank in TARGET_RANKS:
        os.makedirs(os.path.join(TFRECORD_OUTPUT_DIR, rank), exist_ok=True)
    logging.info(f"Ensured directories exist for ranks: {', '.join(TARGET_RANKS)}")

    # --- Load State ---
    processed_files = set()
    if os.path.exists(STATE_FILE):
        logging.info(f"Found state file '{STATE_FILE}'. Loading list of processed files.")
        with open(STATE_FILE, 'r') as f:
            for line in f:
                processed_files.add(line.strip())
    
    logging.info("Finding all SGF files...")
    all_sgf_files = []
    for root, _, files in os.walk(SGF_DIRECTORY):
        for file in files:
            if file.endswith('.sgf'):
                all_sgf_files.append(os.path.join(root, file))
    
    # --- Filter Out Already Processed Files ---
    files_to_process = [f for f in all_sgf_files if f not in processed_files]
    
    logging.info(f"Found {len(all_sgf_files)} total SGF files.")
    logging.info(f"{len(processed_files)} files have already been processed.")
    logging.info(f"{len(files_to_process)} files remaining to process.")

    if not files_to_process:
        logging.info("All files have been processed. Exiting.")
        return

    random.shuffle(files_to_process)
    
    logging.info(f"Starting preprocessing...")
    total_moves_processed = 0
    target_ranks_set = set(TARGET_RANKS)
    
    # Open the state file in append mode to add newly processed files.
    with open(STATE_FILE, 'a') as state_f:
        for i, file_path in enumerate(files_to_process):
            if i > 0 and i % 1000 == 0:
                logging.info(f"  Processed {i}/{len(files_to_process)} new files...")
            
            # Collect all moves from the SGF first to avoid partial writes on error.
            moves_to_write = {}
            for rank, board_state, next_move in process_sgf(file_path, target_ranks_set):
                if rank not in moves_to_write:
                    moves_to_write[rank] = []
                moves_to_write[rank].append((board_state, next_move))

            # If we found valid moves, write them to their respective shards.
            if moves_to_write:
                for rank, moves in moves_to_write.items():
                    shard_index = random.randint(0, NUM_SHARDS - 1)
                    rank_dir = os.path.join(TFRECORD_OUTPUT_DIR, rank)
                    shard_path = os.path.join(rank_dir, f"data.tfrecord-{shard_index:05d}-of-{NUM_SHARDS:05d}")

                    # Use the dedicated append function
                    write_to_tfrecord_append(shard_path, moves)

                # Now that all moves from this file are safely written, log the file as processed.
                total_moves_processed += sum(len(m) for m in moves_to_write.values())
                state_f.write(file_path + '\n')
                state_f.flush() # Ensure it's written to disk immediately
        
    logging.info("\nPreprocessing complete.")
    logging.info(f"Total moves written to TFRecords in this session: {total_moves_processed}")

if __name__ == '__main__':
    main()
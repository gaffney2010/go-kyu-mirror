import os
import glob
import random
import numpy as np
import tensorflow as tf
from sgfmill import sgf, sgf_moves
import logging

# --- GPU Configuration ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Logging Configuration ---
logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("CUDA_VISIBLE_DEVICES set to -1. Forcing CPU execution.")

# --- Configuration ---
SGF_DIRECTORY = 'data/sgfs-by-date'
TFRECORD_OUTPUT_DIR = 'data/tfrecords'
TARGET_RANKS = [f"{i}k" for i in range(1, 16)]
BOARD_SIZE = 19
FILES_PER_SHARD = 500_000  # Process 1M files per shard
STATE_FILE = 'preprocessing_state.txt'

def get_next_shard_number():
    """Determine the next shard number by examining existing shards in the data folder."""
    max_shard = -1
    
    for rank in TARGET_RANKS:
        rank_dir = os.path.join(TFRECORD_OUTPUT_DIR, rank)
        if os.path.exists(rank_dir):
            # Look for existing shard files: data.tfrecord-XXXXX
            pattern = os.path.join(rank_dir, "data.tfrecord-*")
            existing_files = glob.glob(pattern)
            
            for file_path in existing_files:
                # Extract shard number from filename like "data.tfrecord-00001"
                basename = os.path.basename(file_path)
                try:
                    shard_num_str = basename.split('-')[1]
                    shard_num = int(shard_num_str)
                    max_shard = max(max_shard, shard_num)
                except (IndexError, ValueError):
                    continue
    
    return max_shard + 1

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
    Uses incremental updates instead of rebuilding the board from scratch.
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

    # Initialize board arrays
    black = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    white = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for color, move in plays:
        rank_to_record = None
        if color == 'b' and b_is_target:
            rank_to_record = b_rank
        elif color == 'w' and w_is_target:
            rank_to_record = w_rank

        if rank_to_record:
            # Encode move
            if move is None:
                next_move_index = BOARD_SIZE * BOARD_SIZE
            else:
                row, col = move
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    continue
                next_move_index = row * BOARD_SIZE + col

            # Snapshot current state
            board_state = np.stack([black, white], axis=-1).copy()
            yield rank_to_record, board_state, next_move_index

        # Update board after recording
        try:
            if move:
                row, col = move
                if color == 'b':
                    black[row, col] = 1.0
                    white[row, col] = 0.0
                else:
                    white[row, col] = 1.0
                    black[row, col] = 0.0
                board.play(row, col, color)
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
    """Creates a tf.train.Example with raw bytes instead of tf.io.serialize_tensor."""
    feature = {
        'board_state': _bytes_feature(board_state.tobytes()),
        'next_move': _int64_feature(next_move),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

class ShardManager:
    """Manages open TFRecord writers for each rank."""
    
    def __init__(self, shard_number):
        self.shard_number = shard_number
        self.writers = {}
        self.move_counts = {rank: 0 for rank in TARGET_RANKS}
        
        # Create directories and open writers
        for rank in TARGET_RANKS:
            rank_dir = os.path.join(TFRECORD_OUTPUT_DIR, rank)
            os.makedirs(rank_dir, exist_ok=True)
            
            shard_path = os.path.join(rank_dir, f"data.tfrecord-{shard_number:05d}")
            self.writers[rank] = tf.io.TFRecordWriter(shard_path)
            
        logging.info(f"Opened shard {shard_number:05d} for all ranks")
    
    def write_move(self, rank, board_state, next_move):
        """Write a single move to the appropriate rank's shard."""
        try:
            example = serialize_example(board_state, next_move)
            self.writers[rank].write(example)
            self.move_counts[rank] += 1
        except Exception as e:
            logging.error(f"Failed to write move for rank {rank}: {e}")
            raise
    
    def close_all(self):
        """Close all open writers."""
        total_moves = 0
        for rank, writer in self.writers.items():
            writer.close()
            moves = self.move_counts[rank]
            total_moves += moves
            logging.info(f"Closed shard {self.shard_number:05d} for rank {rank}: {moves} moves")
        
        logging.info(f"Shard {self.shard_number:05d} complete: {total_moves} total moves")
        return total_moves

def main():
    """Main function to find, process, and save SGF data to TFRecords."""
    
    # --- Determine starting shard number ---
    current_shard = get_next_shard_number()
    logging.info(f"Starting with shard number: {current_shard:05d}")
    
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
    target_ranks_set = set(TARGET_RANKS)
    
    # Initialize shard manager
    shard_manager = ShardManager(current_shard)
    files_processed_in_shard = 0
    newly_processed_files = []
    total_moves_processed = 0
    
    try:
        for i, file_path in enumerate(files_to_process):
            if i > 0 and i % 1000 == 0:
                logging.info(f"  Processed {i}/{len(files_to_process)} new files in current shard...")
            
            # Process the SGF file and write moves directly to open shards
            file_had_moves = False
            for rank, board_state, next_move in process_sgf(file_path, target_ranks_set):
                shard_manager.write_move(rank, board_state, next_move)
                file_had_moves = True
                total_moves_processed += 1

            newly_processed_files.append(file_path)
            files_processed_in_shard += 1
            
            # Check if we've hit the shard limit
            if files_processed_in_shard >= FILES_PER_SHARD:
                logging.info(f"Shard limit reached ({FILES_PER_SHARD} files). Closing shard {current_shard:05d}")
                
                # Close current shard
                moves_in_shard = shard_manager.close_all()
                
                # Update state file with all newly processed files
                with open(STATE_FILE, 'a') as state_f:
                    for processed_file in newly_processed_files:
                        state_f.write(processed_file + '\n')
                
                logging.info(f"Updated state file with {len(newly_processed_files)} newly processed files")
                
                # Start new shard if there are more files to process
                remaining_files = len(files_to_process) - (i + 1)
                if remaining_files > 0:
                    current_shard += 1
                    shard_manager = ShardManager(current_shard)
                    files_processed_in_shard = 0
                    newly_processed_files = []
                    logging.info(f"Started new shard {current_shard:05d}. {remaining_files} files remaining.")
                else:
                    break
        
        # Handle remaining files if we didn't hit the shard limit
        if newly_processed_files:
            logging.info(f"Processing complete. Closing final shard {current_shard:05d}")
            moves_in_final_shard = shard_manager.close_all()
            
            # Update state file with remaining processed files
            with open(STATE_FILE, 'a') as state_f:
                for processed_file in newly_processed_files:
                    state_f.write(processed_file + '\n')
            
            logging.info(f"Updated state file with final {len(newly_processed_files)} processed files")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        # Make sure to close any open writers
        try:
            shard_manager.close_all()
        except:
            pass
        raise
    
    logging.info(f"\nPreprocessing complete. Total moves processed: {total_moves_processed}")

if __name__ == '__main__':
    main()
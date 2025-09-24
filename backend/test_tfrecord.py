import os
import tensorflow as tf

# Configuration
TFRECORD_DIRECTORY = 'data/tfrecords/10k'
TFRECORD_FILE = 'data.tfrecord-00000'
BOARD_SIZE = 19
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 2)
NUM_OUTPUTS = BOARD_SIZE * BOARD_SIZE + 1

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

def test_tfrecord():
    file_path = os.path.join(TFRECORD_DIRECTORY, TFRECORD_FILE)
    
    print(f"Testing TFRecord file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("ERROR: File is empty!")
        return
    
    # Test both compression types
    for compression_type in [None, "GZIP"]:
        print(f"\n--- Testing compression_type: {compression_type} ---")
        try:
            # Create dataset
            dataset = tf.data.TFRecordDataset(file_path, compression_type=compression_type)
            
            print("Dataset created successfully")
            
            # Try to read first raw record
            print("Attempting to read first raw record...")
            for raw_record in dataset.take(1):
                print(f"Raw record size: {len(raw_record.numpy())} bytes")
                break
            
            # Try to parse first record
            print("Attempting to parse first record...")
            for raw_record in dataset.take(1):
                parsed = parse_tfrecord_fn(raw_record)
                print(f"Parsed successfully!")
                print(f"Board state shape: {parsed[0].shape}")
                print(f"Next move shape: {parsed[1].shape}")
                break
                
            print(f"SUCCESS with compression_type: {compression_type}")
            break
            
        except Exception as e:
            print(f"FAILED with compression_type {compression_type}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_tfrecord()
import numpy as np
import os
import argparse
import glob

from tqdm import tqdm

def split_and_save_array(array_dir, chunk_size, save_dir, save_pattern):
    """
    This function reads numpy arrays from provided directory, combines them into a single array,
    splits the combined array into chunks of a specified size,
    and saves those chunks as separate files in a specified directory.
    """
    # Validate the chunk size
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    # List to hold all arrays
    arrays = []

    # Iterate over each .npy file in the directory
    for path_to_array in glob.glob(os.path.join(array_dir, '*.npy')):
        # Load the array
        array = np.load(path_to_array)
        arrays.append(array)

    # Combine all arrays into one
    combined_array = np.concatenate(arrays, axis=0)
    print(f'Combined array of size {combined_array.shape}')

    # Calculate the number of chunks
    num_chunks = (combined_array.shape[0] + chunk_size - 1) // chunk_size

    # Split the combined array into chunks
    chunks = np.array_split(combined_array, num_chunks)
    print(f'Split combined array into {num_chunks} chunks')

    # Save each chunk to a separate file in the save directory
    for i, chunk in enumerate(tqdm(chunks), start=1):
        save_path = os.path.join(save_dir, f'{save_pattern}_{str(i).zfill(3)}.npy')
        np.save(save_path, chunk)

    print(f'Successfully split the combined array into chunks of size {chunk_size} and saved to directory {save_dir}')


def main():
    parser = argparse.ArgumentParser(description="Split numpy arrays into chunks of a specified size and save them into a directory.")
    parser.add_argument("array_dir", type=str, help="Directory containing the numpy arrays")
    parser.add_argument("chunk_size", type=int, help="Size of each chunk")
    parser.add_argument("save_dir", type=str, help="Directory to save the chunks")
    parser.add_argument("save_pattern", type=str, help="Pattern for saving the chunks")

    args = parser.parse_args()

    split_and_save_array(args.array_dir, args.chunk_size, args.save_dir, args.save_pattern)


if __name__ == "__main__":
    # python script.py array_dir chunk_size save_dir save_pattern
    main()

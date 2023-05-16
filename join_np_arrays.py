import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def get_shape(file_path):
    return np.load(file_path, mmap_mode='r').shape

def load_and_append(args):
    file_path, memmap, start_idx = args
    array = np.load(file_path)
    end_idx = start_idx + len(array)
    memmap[start_idx:end_idx] = array

def main():
    parser = argparse.ArgumentParser(description="Join .npy files into a giant numpy array")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .npy files")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the giant .npy file")
    parser.add_argument("--num_jobs", type=int, default=2, help="Number of jobs to run in parallel")

    args = parser.parse_args()

    npy_files_1 = glob.glob(os.path.join(args.input_dir, '**/*vith14*/**/*.npy'), recursive=True)
    npy_files_2 = glob.glob(os.path.join('/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/img_emb', '*.npy'), recursive=True)
    npy_files = npy_files_1 + npy_files_2

    shapes = Parallel(n_jobs=2)(
        delayed(get_shape)(file_path) for file_path in tqdm(npy_files, desc='Getting shapes')
    )

    # Check all shapes are compatible
    first_shape = shapes[0]
    if not all(shape[1:] == first_shape[1:] for shape in shapes):
        raise ValueError("All .npy files must have the same shape in dimensions other than the first!")

    total_length = sum(shape[0] for shape in shapes)
    dtype = np.load(npy_files[0], mmap_mode='r').dtype

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    print(f"Total length: {total_length}")
    print(f"First shape: {first_shape}")
    print(f"Dtype: {dtype}")
    print(f"Output path: {args.output_path}")
    print(f"{(total_length,) + first_shape[1:]} is the shape of the output array")

    memmap = np.memmap(args.output_path, dtype=dtype, mode='w+', shape=(total_length,) + first_shape[1:])

    Parallel(n_jobs=args.num_jobs)(
        delayed(load_and_append)(args)
        for args in tqdm(
            (
                (file_path, memmap, sum(shape[0] for shape in shapes[:i]))
                for i, file_path in enumerate(npy_files)
            ),
            total=len(npy_files),
            desc='Appending files'
        )
    )

if __name__ == "__main__":
    # python join_np_arrays.py --input_dir /path/to/input/dir --output_path /path/to/output/file.npy
    main()

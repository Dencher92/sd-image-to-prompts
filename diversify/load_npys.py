import os
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from io import BytesIO
import concurrent.futures

# Define a function to process a group
def process_group(filename, group, output_dir_df, output_dir_npy):
    # Generate the URL
    file_id = filename.replace('metadata_', '').replace('.parquet', '')
    url = f'https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/text_emb/text_emb_{file_id}.npy'

    # Download the .npy file into memory
    response = requests.get(url)
    response.raise_for_status()
    data = np.load(BytesIO(response.content))

    # Select the needed rows
    needed_rows = data[group['row_index'].values]

    # Create subdirectories for this file
    output_subdir_df = os.path.join(output_dir_df, file_id)
    output_subdir_npy = os.path.join(output_dir_npy, file_id)
    os.makedirs(output_subdir_df, exist_ok=True)
    os.makedirs(output_subdir_npy, exist_ok=True)

    # Save to .npy files in chunks of 100,000 rows
    for i in tqdm(range(0, len(needed_rows), 100000)):
        chunk = needed_rows[i:i + 100000]
        np.save(os.path.join(output_subdir_npy, f'chunk_{i//100000}.npy'), chunk)

        # Save the corresponding chunk of the DataFrame
        df_chunk = group.iloc[i:i + 100000]
        df_chunk.to_parquet(os.path.join(output_subdir_df, f'chunk_{i//100000}.parquet'))

# Input directory
input_dir = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/metadata_cleaned_5'

# Output directories
output_dir_df = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/meta_filtered'
output_dir_npy = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/text_emb_filtered'

# Load the final DataFrame
print('loading parquet (15Gb takes a while, mkay?)')
df = pd.read_parquet(os.path.join(input_dir, 'final.parquet'))
print(f'Loaded {len(df)} rows')
print(df.columns)
print(f'Loaded {len(df["filename"].unique())} files')


sleep(10)

# Group the DataFrame by filename
grouped = df.groupby('filename')

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_group, filename, group, output_dir_df, output_dir_npy) for filename, group in grouped]

    # Wait for all tasks to complete
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # If a task raised an exception, re-raise it
        future.result()

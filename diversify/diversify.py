from multiprocessing import Pool
import os
import unicodedata
import pandas as pd
from tqdm import tqdm

def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True

def process_batch(files, input_dir, output_dir, batch_no):
    print(f'loading batch {batch_no}...')
    df = pd.DataFrame()

    for file in tqdm(files):
        # Load only the 'caption' column
        data = pd.read_parquet(os.path.join(input_dir, file), columns=['caption'])
        # Add filename and row index
        # data['filename'] = file
        # data['row_index'] = data.index
        df = pd.concat([df, data])

    print('processing captions...')
    # df['caption'] = df['caption'].str.strip()
    # df = df[df['caption'].map(lambda x: len(x.split())) >= 5]
    # df = df[~df['caption'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
    # df = df[df['caption'].apply(is_english_only)]
    df['head'] = df['caption'].str[:15]
    df['tail'] = df['caption'].str[-15:]
    df.drop_duplicates(subset='head', inplace=True)
    df.drop_duplicates(subset='tail', inplace=True)

    # Save the batch to a parquet file
    print('saving batch...')
    df.to_parquet(os.path.join(output_dir, f'batch_{batch_no}.parquet'))

# Input directory
input_dir = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/metadata_cleaned_4'

# Output directory
output_dir = '/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/metadata_cleaned_5'

# Get all the parquet files
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])

# Batch size
batch_size = 4

# Iterate over the files in batches
for i in range(0, len(files), batch_size):
    print(f"Processing batch {i // batch_size + 1}...")
    batch_files = files[i:i + batch_size]
    process_batch(batch_files, input_dir, output_dir, i // batch_size)


# # Multiprocessing version:

# with Pool(4) as p:
#     # i failed and added 5 instead of 6!
#     p.starmap(process_batch, [(files[i:i + batch_size], input_dir, output_dir, i // batch_size + 5) for i in range(0, len(files), batch_size)])
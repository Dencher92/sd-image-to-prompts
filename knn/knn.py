import argparse
import numpy as np
from tqdm import tqdm

from indices import IVFIPIndex


def run_knn(
    query_memmap_path,
    query_memmap_length,
    query_memmap_dim,
    query_memmap_dtype,
    query_start,
    query_end,
    query_batch_size,
    doc_memmap_path,
    doc_memmap_length,
    doc_memmap_dim,
    doc_memmap_dtype,
    doc_start,
    doc_end,
    number_of_partition=8,
    search_in_x_partitions=3,
    use_gpu=True,
    k=5,
):
    print('Load the memmap files')
    query_shape = (query_memmap_length, query_memmap_dim)
    query_memmap = np.memmap(query_memmap_path, dtype=query_memmap_dtype, mode='r', shape=query_shape)

    doc_shape = (doc_memmap_length, doc_memmap_dim)
    doc_memmap = np.memmap(doc_memmap_path, dtype=doc_memmap_dtype, mode='r', shape=doc_shape)
    doc_memmap = doc_memmap[doc_start:doc_end]

    # now you should be careful with dtypes, because index uses inner product as metric, therefore we have to
    # use signed int16 as dtype. so your query and doc memmap should be signed int16 as well or should not exceed
    # positive int16 range.

    print('Build the index')
    index = IVFIPIndex(doc_memmap)
    index.build(
        number_of_partition=number_of_partition,
        search_in_x_partitions=search_in_x_partitions,
        use_gpu=use_gpu
    )

    similarities = []
    indices = []
    for i in tqdm(range(query_start, query_end, query_batch_size), desc='Querying'):
        batch = query_memmap[i:i + query_batch_size]
        batch_distances, batch_indices = index.query(batch, k=k)
        similarities.append(batch_distances)
        indices.append(batch_indices)

    similarities = np.concatenate(similarities)
    indices = np.concatenate(indices)

    return similarities, indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_memmap_path", type=str, help="Path to the memmap file")
    parser.add_argument("--query_memmap_length", type=int, help="Length of the memmap file")
    parser.add_argument("--query_memmap_dim", type=int, help="Dimension of the memmap file")
    parser.add_argument("--query_memmap_dtype", type=str, help="Dtype of the memmap file")
    parser.add_argument("--query_batch_size", type=int, default=1000, help="Batch size for querying")

    parser.add_argument("--doc_memmap_path", type=str, help="Path to the memmap file")
    parser.add_argument("--doc_memmap_length", type=int, help="Length of the memmap file")
    parser.add_argument("--doc_memmap_dim", type=int, help="Dimension of the memmap file")
    parser.add_argument("--doc_memmap_dtype", type=str, help="Dtype of the memmap file")

    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")

    args = parser.parse_args()
    run_knn(**vars(args))
    print('done!')

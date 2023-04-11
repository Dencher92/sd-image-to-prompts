import argparse
import numpy as np
import pandas as pd

from knn import run_knn


def run_batched_knn(args):

    similarities, rel_ids, abs_ids = [], [], []

    for i in range(args.doc_start, args.doc_end, args.doc_batch_size):
        print(f'Loading doc batch {i} to {i + args.doc_batch_size}')
        similarities_batch, ids_batch = run_knn(
            query_memmap_path=args.query_memmap_path,
            query_memmap_length=args.query_memmap_length,
            query_memmap_dim=args.query_memmap_dim,
            query_memmap_dtype=args.query_memmap_dtype,
            query_start=args.query_start,
            query_end=args.query_end,
            query_batch_size=args.query_batch_size,
            doc_memmap_path=args.doc_memmap_path,
            doc_memmap_length=args.doc_memmap_length,
            doc_memmap_dim=args.doc_memmap_dim,
            doc_memmap_dtype=args.doc_memmap_dtype,
            doc_start=i,
            doc_end=i + args.doc_batch_size,
            use_gpu=args.use_gpu,
            k=args.k,
        )
        similarities.append(similarities_batch)
        rel_ids.append(ids_batch)
        abs_ids.append(ids_batch + i)

    similarities = np.concatenate(similarities, axis=1)
    rel_ids = np.concatenate(rel_ids, axis=1)
    abs_ids = np.concatenate(abs_ids, axis=1)

    # sort:
    sort_indices = np.argsort(-similarities, axis=1)
    sorted_distances = np.take_along_axis(similarities, sort_indices, axis=1)
    sorted_abs_ids = np.take_along_axis(abs_ids, sort_indices, axis=1)

    # clip
    sorted_distances_clipped = sorted_distances[:, :args.k]
    sorted_abs_ids_clipped = sorted_abs_ids[:, :args.k]

    # make a dataframe with results:
    res = pd.DataFrame({
        'query_id': np.arange(args.query_start, args.query_end),
        'doc_ids': sorted_abs_ids_clipped.tolist(),
        'distances': sorted_distances_clipped.tolist(),
    })

    res.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_memmap_path", type=str, help="Path to the memmap file")
    parser.add_argument("--query_memmap_length", type=int, help="Length of the memmap file")
    parser.add_argument("--query_memmap_dim", type=int, help="Dimension of the memmap file")
    parser.add_argument("--query_memmap_dtype", type=str, help="Dtype of the memmap file")
    parser.add_argument("--query_batch_size", type=int, default=1000, help="Batch size for querying")
    parser.add_argument("--query_start", type=int, default=0, help="Start index for querying")
    parser.add_argument("--query_end", type=int, default=1000, help="End index for querying")

    parser.add_argument("--doc_memmap_path", type=str, help="Path to the memmap file")
    parser.add_argument("--doc_memmap_length", type=int, help="Length of the memmap file")
    parser.add_argument("--doc_memmap_dim", type=int, help="Dimension of the memmap file")
    parser.add_argument("--doc_memmap_dtype", type=str, help="Dtype of the memmap file")
    parser.add_argument("--doc_start", type=int, default=0, help="Start index for querying")
    parser.add_argument("--doc_end", type=int, default=1000, help="End index for querying")
    parser.add_argument("--doc_batch_size", type=int, default=1_000_000, help="Batch size for querying")

    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")

    parser.add_argument("--output_path", type=str, help="Path to the output file")

    args = parser.parse_args()
    run_batched_knn(args)
    print('done!')


# TODO: написать лаунчер, который будет запускать скрипт батчами для докуметов.
# При этом он будет собирать в памяти результаты (q_id, nn_id, dist) с каждого батча.
# После этого он будет сортировать результаты по q_id и записывать в файл, например
# Так же он будет передавать id документов в кнн, т.к. батчи будут иметь каждый свой оффсет

import faiss


# K-means on dataset, no vector-quantization
class IVFIndex():
    def __init__(self, vectors):
        self.dimension = vectors.shape[1]
        self.vectors = vectors

    def build(
        self,
        number_of_partition=8,
        search_in_x_partitions=2,
    ):
        quantizer = faiss.IndexFlatL2(self.dimension)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            n_list=number_of_partition,
            n_probe=search_in_x_partitions
        )
        self.index.train(self.vectors)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        return distances, indices


# K-means on dataset and vector-quantization
class IVPQIndex():
    def __init__(self, vectors):
        self.dimension = vectors.shape[1]
        self.vectors = vectors

    def build(
        self,
        number_of_partition=8,
        search_in_x_partitions=2,
        subvector_size=8,
        use_gpu=True
    ):
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            number_of_partition,
            search_in_x_partitions,
            subvector_size
        )

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.index = index

        self.index.train(self.vectors)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        return distances, indices


# this one works with integer vectors because it uses the inner product, which
# does not involve any floating point operations
class IVFIPIndex():
    def __init__(self, vectors):
        self.dimension = vectors.shape[1]
        self.vectors = vectors

    def build(
        self,
        number_of_partition=8,
        search_in_x_partitions=2,
        use_gpu=True
    ):
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            number_of_partition,
        )

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.index = index

        self.index.train(self.vectors)
        self.index.add(self.vectors)
        self.index.nprobe = search_in_x_partitions

    def query(self, vectors, k=10):
        # because we use the inner product, we need to invert the vectors:
        vectors = -vectors
        # also simple dot product will facilitate vectors with big values, so wee need to normalize it:
        # faiss.normalize_L2(vectors)
        D, I = self.index.search(vectors, k)
        return D, I


# There is also a variant of quantization that encodes DOC vectors as distances to centroids,
# rather than as indices of centroids. This is called Inverted Multi-Index (IVF) quantization.

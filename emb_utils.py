import faiss
import numpy as np

class EmbeddingIndex(object):
    def __init__(self, dimension, index_params=None):
        self.dimension = dimension
        self.index_params = index_params if index_params else {}
        self.index = self._create_index()
        self.metadata = []

    def _create_index(self):
        index_type = self.index_params.get('type', 'FlatL2')

        if index_type == 'FlatL2':
            return faiss.IndexFlatL2(self.dimension)
        elif index_type == 'FlatIP':
            return faiss.IndexFlatIP(self.dimension)
        elif index_type == 'IVFFlat':
            nlist = self.index_params.get('nlist', 100)
            return faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dimension), self.dimension, nlist)
        elif index_type == 'IVFPQ':
            nlist = self.index_params.get('nlist', 100)
            m = self.index_params.get('m', 8)
            nbits = self.index_params.get('nbits', 8)
            return faiss.IndexIVFPQ(faiss.IndexFlatL2(self.dimension), self.dimension, nlist, m, nbits)
        elif index_type == 'HNSWFlat':
            return faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_L2)
        elif index_type == 'LSH':
            return faiss.IndexLSH(self.dimension, self.index_params.get('hash_size', 10))
        elif index_type == 'BinaryFlat':
            return faiss.IndexBinaryFlat(self.dimension)
        elif index_type == 'ScalarQuantizer':
            return faiss.IndexScalarQuantizer(self.dimension, faiss.ScalarQuantizer.QT_8bit)

    def add_embeddings(self, embeddings, metadata):
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embeddings of dimension {self.dimension}, got {embeddings.shape[1]}")

        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings and metadata entries should be the same")

        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)

    def search(self, query_embedding, k=5):
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected query embedding of dimension {self.dimension}, got {query_embedding.shape[1]}")

        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return distances, indices, [self.metadata[i] for i in indices[0]]

    def save_index(self, filename):
        faiss.write_index(self.index, filename)
        np.savez_compressed(f"{filename}_metadata.npz", metadata=self.metadata)

    def load_index(self, filename):
        self.index = faiss.read_index(filename)
        metadata_file = np.load(f"{filename}_metadata.npz")
        self.metadata = metadata_file['metadata']
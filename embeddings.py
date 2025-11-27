import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

def load_repo_embeddings(path="models_cache/repo_embeddings.npz"):
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["embeds"]

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

REPO_PATH = "data/repository.json"
EMB_PATH = "models_cache/repo_embeddings.npz"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("Ouverture du référentiel...")
    with open(REPO_PATH, "r") as f:
        repo = json.load(f)

    competencies = repo["competencies"]
    comp_ids = list(competencies.keys())
    comp_texts = [competencies[cid] for cid in comp_ids]

    print(f"Encodage de {len(comp_texts)} compétences avec SBERT...")
    model = SentenceTransformer(MODEL_NAME)
    embeds = model.encode(comp_texts, convert_to_numpy=True, show_progress_bar=True)
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    print("Sauvegarde des embeddings…")
    np.savez_compressed(EMB_PATH, ids=np.array(comp_ids), embeds=embeds)
    print(f"Embeddings saved - {EMB_PATH}")

if __name__ == "__main__":
    main()

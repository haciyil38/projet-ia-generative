import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import sys
import os
sys.path.append(os.path.abspath(".."))
from embeddings import load_repo_embeddings

# Chargement référentiel et embeddings
REPO_EMB_PATH = "../models_cache/repo_embeddings.npz"
REPO_JSON_PATH = "../data/repository.json"

repo_ids, repo_embeds = load_repo_embeddings(REPO_EMB_PATH)

with open(REPO_JSON_PATH, "r") as f:
    repo = json.load(f)

# Inputs utilisateur à scorer (tu peux tester avec plusieurs exemples)
user_inputs = [
    "J’utilise Python pour analyser et nettoyer des données.",
    "Je sais faire un modèle de régression pour un projet."
]

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
user_embs = model.encode(user_inputs, convert_to_numpy=True)

# Calcul des similarités cosinus
similarities = util.cos_sim(user_embs, repo_embeds).cpu().numpy()

# ---- Brique clé : score moyen par compétence -> score moyen par bloc ----
block_scores = {}

for block in repo["blocks"]:
    block_name = block["name"]
    comp_ids = block["competencies"]
    # Récupère les indices correspondants dans repo_ids
    idxs = [repo_ids.index(cid) for cid in comp_ids]
    # Pour chaque compétence du bloc, on prend le max(score) sur les phrases utilisateur
    comp_max_scores = similarities[:, idxs].max(axis=0)
    # Score du bloc : moyenne des max de chaque compétence
    block_scores[block_name] = float(np.mean(comp_max_scores))

print("----- Score par bloc de compétences -----")
for b, s in block_scores.items():
    print(f"{b:20} : {s:.2f}")

# ---- Proposition de matching métiers suivant seuil ----
print("\n----- Recommandation métiers -----")
for job_id, job in repo["jobs"].items():
    # Pour chaque compétence requise, récupérer son bloc et son score
    scores = []
    for req in job["requirements"]:
        # Trouve le bloc contenant cette compétence
        for block in repo["blocks"]:
            if req in block["competencies"]:
                bscore = block_scores[block["name"]]
                scores.append(bscore)
                break
    if scores:  # Si le métier nécessite au moins 1 bloc scoré
        job_score = np.mean(scores)
        print(f"{job['title']:20} : score global {job_score:.2f}")

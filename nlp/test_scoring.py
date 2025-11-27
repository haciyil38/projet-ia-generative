import numpy as np
from sentence_transformers import SentenceTransformer, util
from embeddings import load_repo_embeddings

# Chargement des embeddings référentiel
REPO_EMB_PATH = "../models_cache/repo_embeddings.npz"  # Ajuste ce chemin selon où tu lances ce script
repo_ids, repo_embeds = load_repo_embeddings(REPO_EMB_PATH)

# Exemple de réponses utilisateur (à remplacer par tes propres inputs)
user_inputs = [
    "J’utilise Python pour analyser et nettoyer des données.",
    "Je sais faire un modèle de régression pour un projet."
]

# Encodage des phrases utilisateur avec SBERT local
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
user_embs = model.encode(user_inputs, convert_to_numpy=True)

# Calcul de la matrice des similarités cosinus
similarities = util.cos_sim(user_embs, repo_embeds).cpu().numpy()

# Affichage détaillé du mapping utilisateur <-> compétences
print("Réponses utilisateur :")
for i, sent in enumerate(user_inputs):
    print(f"[{i:1}] {sent}")

print("\nCompétences du référentiel :")
for i, cid in enumerate(repo_ids):
    print(f"[{i:1}] {cid}")

print("\nMatrice similarité (ligne=réponse utilisateur, colonne=compétence) :")
print(np.round(similarities, 2))

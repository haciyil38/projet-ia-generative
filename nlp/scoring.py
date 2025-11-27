import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import os

MODEL_NAME = "all-MiniLM-L6-v2"

class ScoringEngine:
    def __init__(self, repo_path="data/repository.json", embeddings_path="models_cache/repo_embeddings.npz"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.repo = self._load_repo(repo_path)
        self.repo_ids, self.repo_embeddings = self._load_embeddings(embeddings_path)
        
        # Map ID to index for quick lookup
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.repo_ids)}

    def _load_repo(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _load_embeddings(self, path):
        data = np.load(path, allow_pickle=True)
        return list(data["ids"]), data["embeds"]

    def encode_user_input(self, user_texts):
        """
        Encodes a list of user text inputs into embeddings.
        """
        return self.model.encode(user_texts, convert_to_numpy=True)

    def calculate_similarity(self, user_embeddings):
        """
        Calculates the cosine similarity between user embeddings and repository embeddings.
        Returns a matrix of shape (n_user_inputs, n_repo_competencies).
        """
        return util.cos_sim(user_embeddings, self.repo_embeddings).numpy()

    def compute_scores(self, user_texts):
        """
        Computes block scores and job matches based on user text inputs.
        """
        if not user_texts:
            return {}, []

        user_embeddings = self.encode_user_input(user_texts)
        sim_matrix = self.calculate_similarity(user_embeddings)
        
        # sim_matrix shape: (num_inputs, num_competencies)
        # We want the MAX similarity for each competency across all user inputs
        # i.e., if user mentions Python in one sentence, that counts for the Python competency
        max_scores_per_competency = np.max(sim_matrix, axis=0)
        
        # Create a mapping of Competency ID -> Score
        comp_scores = {cid: float(max_scores_per_competency[self.id_to_idx[cid]]) for cid in self.repo_ids}

        # 1. Calculate Block Scores
        block_scores = {}
        for block in self.repo["blocks"]:
            b_id = block["id"]
            b_name = block["name"]
            c_ids = block["competencies"]
            
            # Average score of competencies in this block
            # We can also use max or weighted average
            scores = [comp_scores.get(cid, 0.0) for cid in c_ids]
            avg_score = np.mean(scores) if scores else 0.0
            block_scores[b_name] = round(float(avg_score), 2)

        # 2. Calculate Job Matches
        job_matches = []
        for j_id, job in self.repo["jobs"].items():
            reqs = job["requirements"]
            # Average score of required competencies
            scores = [comp_scores.get(cid, 0.0) for cid in reqs]
            match_score = np.mean(scores) if scores else 0.0
            
            job_matches.append({
                "job_id": j_id,
                "title": job["title"],
                "score": round(float(match_score), 2),
                "missing_skills": [cid for cid, s in zip(reqs, scores) if s < 0.5] # Threshold for missing
            })
        
        # Sort jobs by score descending
        job_matches.sort(key=lambda x: x["score"], reverse=True)

        return block_scores, job_matches

if __name__ == "__main__":
    # Simple test
    engine = ScoringEngine()
    inputs = [
        "I have experience cleaning data with Python and building dashboards.",
        "I studied regression models and deep learning."
    ]
    blocks, jobs = engine.compute_scores(inputs)
    print("Block Scores:", blocks)
    print("Top Jobs:", jobs[:3])

import google.generativeai as genai
import json
import os
import hashlib
import streamlit as st

# Simple JSON Cache
CACHE_FILE = "models_cache/genai_cache.json"

class GenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            # Try to get from streamlit secrets if available
            try:
                self.api_key = st.secrets["general"]["GEMINI_API_KEY"]
            except:
                pass
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
            print("Warning: No API Key found for Gemini.")

        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, prompt):
        return hashlib.md5(prompt.encode()).hexdigest()

    def generate_content(self, prompt):
        key = self._get_cache_key(prompt)
        if key in self.cache:
            print("Cache hit!")
            return self.cache[key]

        if not self.model:
            return "Error: API Key not configured."

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            self.cache[key] = text
            self._save_cache()
            return text
        except Exception as e:
            return f"Error generating content: {str(e)}"

    def enrich_user_input(self, text):
        """
        Enriches short user input with more context.
        """
        if len(text.split()) >= 5:
            return text
            
        prompt = f"""
        You are an expert technical recruiter. The user provided a very short description of their skills: "{text}".
        Please rewrite this into a single, concise professional sentence that expands on the implied skills (e.g. if they say "python", mention "Python programming for data analysis or scripting").
        Do not invent skills they didn't mention, just contextualize them.
        Output ONLY the rewritten sentence.
        """
        return self.generate_content(prompt).strip()

    def generate_progression_plan(self, job_title, missing_skills, block_scores):
        """
        Generates a learning plan.
        """
        prompt = f"""
        You are a career coach. The user wants to become a "{job_title}".
        
        Their current skill coverage:
        {json.dumps(block_scores, indent=2)}
        
        Missing specific skills for this role:
        {', '.join(missing_skills)}
        
        Please generate a concise 3-step progression plan to close these gaps.
        Format as Markdown.
        """
        return self.generate_content(prompt)

    def generate_bio(self, top_skills, best_job_match):
        """
        Generates a professional bio.
        """
        prompt = f"""
        Write a short (2-3 sentences) professional bio for a LinkedIn profile.
        
        Top Skills: {', '.join(top_skills)}
        Target Role: {best_job_match}
        
        Tone: Professional, ambitious, and concise.
        """
        return self.generate_content(prompt)

# AISCA: Agent Intelligent Sémantique et Génératif

AISCA is a "Mini-Agent RAG" designed to map user skills to job profiles using Semantic Analysis (SBERT) and Generative AI (Google Gemini).

## Features
- **Semantic Scoring**: Uses `sentence-transformers` to calculate cosine similarity between user inputs and a competency repository.
- **Job Recommendation**: Matches users to roles like Data Analyst, ML Engineer, or Data Scientist.
- **GenAI Insights**:
    - **Smart Enrichment**: Expands short user answers to provide better context.
    - **Progression Plan**: Generates a personalized learning path for missing skills.
    - **Professional Bio**: Writes a LinkedIn-style summary.
- **Cost-Efficient**: Implements local caching for API calls to minimize costs.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd aisca
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup API Key**:
   - Copy the secrets template:
     ```bash
     mkdir -p .streamlit
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```
   - Edit `.streamlit/secrets.toml` and add your Google Gemini API Key.

5. **Initialize Data**:
   ```bash
   python encode_repository.py
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Architecture
- `nlp/scoring.py`: Core logic for embeddings and similarity calculation.
- `genai/client.py`: Wrapper for Gemini API with caching.
- `data/repository.json`: JSON database of skills and jobs.
- `app.py`: Streamlit frontend.

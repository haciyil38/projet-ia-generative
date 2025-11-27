import streamlit as st
import pandas as pd
import plotly.express as px
from nlp.scoring import ScoringEngine
from genai.client import GenAIClient

# Page Config
st.set_page_config(page_title="AISCA - Skills Mapper", layout="wide")

# Initialize Engines
@st.cache_resource
def load_engines():
    return ScoringEngine(), GenAIClient()

scoring_engine, genai_client = load_engines()

st.title("🧠 AISCA: Intelligent Skills Mapping Agent")
st.markdown("Analyze your skills, get job recommendations, and receive a personalized AI progression plan.")

# --- Sidebar: User Inputs ---
st.sidebar.header("Your Profile")

# 1. Self-Assessment (Likert)
st.sidebar.subheader("Self-Assessment (1-5)")
level_python = st.sidebar.slider("Python Programming", 1, 5, 3)
level_ml = st.sidebar.slider("Machine Learning Concepts", 1, 5, 2)
level_data = st.sidebar.slider("Data Analysis & Viz", 1, 5, 3)

# 2. Open Text Inputs
st.sidebar.subheader("Experience Details")
text_projects = st.sidebar.text_area("Describe your projects & technical skills:", 
                                     "I have used Python for data cleaning and created dashboards using Streamlit. I also know basic regression models.")

text_education = st.sidebar.text_area("Education & Certifications:", 
                                      "I have a degree in Computer Science and studied deep learning basics.")

if st.sidebar.button("Analyze Profile 🚀"):
    with st.spinner("Analyzing semantics and generating insights..."):
        
        # --- 1. NLP Analysis ---
        # Combine inputs
        user_inputs = [text_projects, text_education]
        
        # Enrich if short (using GenAI)
        enriched_inputs = []
        for text in user_inputs:
            if len(text.split()) < 5:
                # Only enrich if really short to save calls, though logic is in client
                enriched_text = genai_client.enrich_user_input(text)
                enriched_inputs.append(enriched_text)
            else:
                enriched_inputs.append(text)
        
        # Compute Scores
        block_scores, job_matches = scoring_engine.compute_scores(enriched_inputs)
        
        # --- 2. Display Results ---
        
        # Layout: 2 Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Skills Coverage")
            # Radar Chart
            df_scores = pd.DataFrame(dict(
                r=list(block_scores.values()),
                theta=list(block_scores.keys())
            ))
            fig = px.line_polar(df_scores, r='r', theta='theta', line_close=True, range_r=[0,1])
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Top Match:** {job_matches[0]['title']} ({int(job_matches[0]['score']*100)}%)")

        with col2:
            st.subheader("💼 Job Recommendations")
            for job in job_matches[:3]:
                with st.expander(f"{job['title']} - {int(job['score']*100)}% Match", expanded=(job==job_matches[0])):
                    st.write(f"**Missing Skills:** {', '.join(job['missing_skills']) if job['missing_skills'] else 'None! Great fit.'}")
                    if st.button(f"Generate Plan for {job['title']}", key=job['job_id']):
                        plan = genai_client.generate_progression_plan(job['title'], job['missing_skills'], block_scores)
                        st.markdown(plan)

        # --- 3. GenAI Insights ---
        st.markdown("---")
        st.subheader("✨ AI Personal Branding")
        
        if genai_client.api_key:
            bio = genai_client.generate_bio(list(block_scores.keys()), job_matches[0]['title'])
            st.success("**Professional Bio:**")
            st.write(bio)
        else:
            st.warning("GenAI features (Bio, Progression Plan) require an API Key. Please add it to `.streamlit/secrets.toml`.")

else:
    st.info("Fill out the sidebar and click 'Analyze Profile' to start.")

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIResumeScreeningSystem:
    def __init__(self, target_skills):
        """
        Initializes the screening pipeline with a specified list of core tracking competencies.
        Converts all targets to lowercase and configures a TF-IDF vectorizer utilizing
        unigrams and bigrams, applying sublinear scaling to penalize excessive term repetition.
        """
        self.target_skills = [skill.lower() for skill in target_skills]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    def clean_text(self, text):
        """
        Performs structural text sanitation on unstructured strings.
        Removes HTML tags, email addresses, absolute URLs, and special characters, 
        returning a normalized, lowercase single-spaced token string.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<[^>]*>', ' ', text)  
        text = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', text)  
        text = re.sub(r'http\S+|www\S+', ' ', text)  
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    def extract_skills(self, cleaned_text):
        """
        Executes explicit regex token matching against a list of pre-defined target competencies.
        Enforces word boundaries (\b) to isolate targeted keywords from nested substrings,
        sorting identified metrics into sets of matched and missing categories.
        """
        matched = []
        missing = []
        for skill in self.target_skills:
            pattern = rf'\b{re.escape(skill)}\b'
            if re.search(pattern, cleaned_text):
                matched.append(skill.upper())
            else:
                missing.append(skill.upper())
        return ", ".join(matched), ", ".join(missing)

    def process_and_rank(self, df, job_description):
        """
        Orchestrates the primary analysis pipeline.
        Applies data sanitation, fits the TF-IDF feature space, computes the geometric cosine
        similarity score, maps skill gap dependencies, and returns a sorted ranking table.
        """
        processed_df = df.copy()
        
        # Standardize schema names internally to prevent structural KeyError exceptions
        column_mapping = {}
        resume_options = ['resume', 'resume_text', 'resumes', 'resume_str', 'text', 'resume_str']
        category_options = ['category', 'job_category', 'role', 'label', 'class']
        
        for col in processed_df.columns:
            if col.lower() in resume_options:
                column_mapping[col] = 'Resume'
            elif col.lower() in category_options:
                column_mapping[col] = 'Category'
                
        if column_mapping:
            processed_df = processed_df.rename(columns=column_mapping)
            
        # Verify presence of localized data parameters
        if 'Resume' not in processed_df.columns:
            raise KeyError(f"Could not locate text column. Source data columns: {list(df.columns)}")
        if 'Category' not in processed_df.columns:
            processed_df['Category'] = 'Unspecified'
        
        # Step 1: Sanitation Pass
        processed_df['cleaned_resume_text'] = processed_df['Resume'].apply(self.clean_text)
        cleaned_jd = self.clean_text(job_description)
        
        # Step 2: TF-IDF Vector Space & Coordinate Construction
        corpus = processed_df['cleaned_resume_text'].tolist() + [cleaned_jd]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        resume_vectors = tfidf_matrix[:-1]
        jd_vector = tfidf_matrix[-1]
        
        # Step 3: Compute Geometric Alignment Percentage
        similarity_scores = cosine_similarity(resume_vectors, jd_vector).flatten()
        processed_df['match_score_pct'] = np.round(similarity_scores * 100, 2)
        
        # Step 4: Map Overlapping Competencies and Gap Inefficiencies
        skills_audit = processed_df['cleaned_resume_text'].apply(self.extract_skills)
        processed_df['Skills Matched'] = [audit[0] for audit in skills_audit]
        processed_df['Skills Missing'] = [audit[1] for audit in skills_audit]
        
        # Step 5: Isolate Target Output Vectors Ordered by Performance Score
        final_reporting_df = processed_df[[
            'Category', 'match_score_pct', 'Skills Matched', 'Skills Missing'
        ]].sort_values(by='match_score_pct', ascending=False).reset_index(drop=True)
        
        return final_reporting_df

# ==========================================
# SYSTEM PIPELINE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Define mandatory benchmarking keywords used for matrix evaluation
    mandatory_job_skills = ["Python", "SQL", "NLP", "Machine Learning", "Tableau"]
    
    sample_job_description = """
    We are seeking a Machine Learning Engineer with strong hands-on experience in Python and SQL. 
    The ideal candidate should have expertise in Natural Language Processing (NLP), predictive modeling, 
    and building data visualization dashboards using Tableau.
    """
    
    # Ingest target data source file containing the resume records
    dataset = pd.read_csv("Resume.csv") 
    print(f"Loaded dataset containing {len(dataset)} profile records.")

    # Initialize the engine instance and run analytics
    screening_engine = AIResumeScreeningSystem(target_skills=mandatory_job_skills)
    ranked_candidates = screening_engine.process_and_rank(dataset, sample_job_description)
    
    # Display top 10 highest-ranking records to terminal console
    print("\n--- CANDIDATE COMPLIANCE REPORT ---")
    print(ranked_candidates.head(10).to_string())
    
    # Export parsed results to a persistent file location for external review
    ranked_candidates.to_csv("Screening_Results.csv", index=False)
    print("\nResults successfully exported to 'Screening_Results.csv'")
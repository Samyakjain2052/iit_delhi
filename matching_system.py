from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util
import torch
import re
from datetime import datetime
import json
import requests
import pickle

class MatchingSystem:
    def __init__(self, huggingface_token, api_token):
        """Initialize the matching system"""
        self.model = self._initialize_model(huggingface_token)
        self.api_token = api_token
        self.base_url = "https://iit-api-l95f.onrender.com"
        self.job_classifier, self.tfidf = self._load_job_predictor()
    
    def _initialize_model(self, token):
        """Initialize the sentence transformer model"""
        login(token)
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def get_api_data(self):
        """Fetch data from API endpoints"""
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

        try:
            # Get resumes
            resume_response = requests.get(
                f"{self.base_url}/api/resumes",
                headers=headers
            )
            resume_response.raise_for_status()
            resumes = resume_response.json()

            # Get job descriptions
            job_response = requests.get(
                f"{self.base_url}/api/job-descriptions",
                headers=headers
            )
            job_response.raise_for_status()
            jobs = job_response.json()

            return resumes, jobs

        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return [], []

    def calculate_skills_match(self, resume_skills, job_skills):
        """Calculate similarity between resume skills and job requirements"""
        if not resume_skills or not job_skills:
            return 0.0
        
        resume_embeddings = self.model.encode(resume_skills, convert_to_tensor=True)
        job_embeddings = self.model.encode(job_skills, convert_to_tensor=True)
        
        similarity_matrix = util.cos_sim(resume_embeddings, job_embeddings)
        return float(torch.mean(torch.max(similarity_matrix, dim=1)[0]))

    def parse_experience(self, experience_text):
        """Extract years of experience from text"""
        if not experience_text:
            return 0
            
        years_pattern = r'\((\d{4})\s*-\s*(\d{4})\)'
        match = re.search(years_pattern, experience_text)
        
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            current_year = datetime.now().year
            
            if end_year > current_year:
                years = current_year - start_year
            else:
                years = end_year - start_year
            return max(0, years)
        return 0

    def calculate_experience_match(self, job_min_years, resume_experience):
        """Calculate experience match score"""
        if not resume_experience:
            return 0.0
            
        resume_years = self.parse_experience(resume_experience[0])
        
        try:
            min_years = int(job_min_years) if job_min_years != "Not specified" else 0
            if resume_years < min_years:
                return max(0, resume_years / min_years)
            return 1.0
        except (ValueError, TypeError):
            return 0.0

    def calculate_education_match(self, resume_education, job_education):
        """Calculate education match score"""
        if not resume_education or not job_education:
            print("Missing education data")
            return 0.0
        
        resume_edu = " ".join(resume_education) if isinstance(resume_education, list) else str(resume_education)
        job_min_degree = job_education.get('minimum_degree', '')
        job_pref_fields = job_education.get('preferred_fields', [])
        
        # Create embeddings
        resume_embedding = self.model.encode([resume_edu], convert_to_tensor=True)
        job_embedding = self.model.encode([job_min_degree] + job_pref_fields, convert_to_tensor=True)
        
        # Calculate similarity
        similarity = util.cos_sim(resume_embedding, job_embedding)
        return float(torch.max(similarity))

    def _load_job_predictor(self):
        """Load the job type prediction model"""
        try:
            with open("model.pkl", "rb") as model_file:
                model = pickle.load(model_file)
            with open("tfidf.pkl", "rb") as tfidf_file:
                tfidf = pickle.load(tfidf_file)
            return model, tfidf
        except FileNotFoundError:
            print("Warning: Job prediction model files not found")
            return None, None

    def predict_job_type(self, job_title):
        """Predict if a job is technical or non-technical"""
        if not self.job_classifier or not self.tfidf:
            return "Unknown"
        job_vectorized = self.tfidf.transform([job_title])
        prediction = self.job_classifier.predict(job_vectorized)[0]
        return "Technical" if prediction == 1 else "Non-Technical"

    def calculate_project_match(self, resume_projects, job_title, required_skills):
        """Calculate project relevance score based on job type"""
        if not resume_projects:
            return 0.0

        job_type = self.predict_job_type(job_title)
        
        if job_type == "Technical":
            return self._calculate_technical_project_match(resume_projects, required_skills)
        else:
            return self._calculate_non_technical_project_match(resume_projects)

    def _calculate_technical_project_match(self, projects, required_skills):
        """Calculate match score for technical projects"""
        project_info = []
        
        for project in projects:
            # Handle both string and dictionary project formats
            if isinstance(project, dict):
                technologies = project.get('technologies', [])
                description = project.get('description', '')
                project_info.extend(technologies)
                if description:
                    project_info.append(description)
            elif isinstance(project, str):
                # If project is a string, treat it as description
                project_info.append(project)
        
        if not project_info or not required_skills:
            return 0.0
        
        project_embeddings = self.model.encode(project_info, convert_to_tensor=True)
        skills_embeddings = self.model.encode(required_skills, convert_to_tensor=True)
        
        similarity_matrix = util.cos_sim(project_embeddings, skills_embeddings)
        return float(torch.mean(torch.max(similarity_matrix, dim=1)[0]))

    def _calculate_non_technical_project_match(self, projects):
        """Calculate match score for non-technical projects"""
        project_descriptions = []
        
        for project in projects:
            if isinstance(project, dict):
                desc = project.get('description', '')
                if desc:
                    project_descriptions.append(desc)
            elif isinstance(project, str):
                project_descriptions.append(project)
        
        if not project_descriptions:
            return 0.0
        
        # Use general soft skills keywords for non-technical roles
        soft_skills = [
            "communication", "leadership", "teamwork",
            "organization", "management", "coordination"
        ]
        
        project_embeddings = self.model.encode(project_descriptions, convert_to_tensor=True)
        skills_embeddings = self.model.encode(soft_skills, convert_to_tensor=True)
        
        similarity_matrix = util.cos_sim(project_embeddings, skills_embeddings)
        return float(torch.mean(torch.max(similarity_matrix, dim=1)[0]))

    def get_weights_from_db(self):
        """Fetch weights from the database"""
        try:
            response = requests.get(
                f"{self.base_url}/api/job-descriptions",  # Update endpoint if needed
                headers={"Authorization": f"Bearer {self.api_token}"}
            )
            response.raise_for_status()
            data = response.json()
            
            # If API returns a list, use the first item
            weights = data[0] if isinstance(data, list) else data
            
            # Validate and return weights
            return {
                'skills_weight': float(weights.get('skills_weight', 0.30)),
                'experience_weight': float(weights.get('experience_weight', 0.25)),
                'education_weight': float(weights.get('education_weight', 0.25)),
                'project_weight': float(weights.get('project_weight', 0.20))
            }
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Could not fetch weights from DB: {e}")
            # Return default weights if DB fetch fails
            return {
                'skills_weight': 0.30,
                'experience_weight': 0.25,
                'education_weight': 0.25,
                'project_weight': 0.20
            }

    def calculate_matches(self):
        """Calculate comprehensive match scores using weights from DB"""
        resumes, jobs = self.get_api_data()
        weights = self.get_weights_from_db()
        matches = []
        
        for resume in resumes:
            resume_name = resume.get('name', 'Unknown Candidate')
            
            for job in jobs:
                job_title = job.get('title', 'Unknown Position')
                required_skills = job.get('requiredSkills', [])
                
                # Calculate individual scores
                skills_score = self.calculate_skills_match(
                    resume.get('technicalSkills', []),
                    required_skills
                )
                
                # Handle experience requirements safely
                exp_requirements = job.get('experienceRequirements', 'Not specified')
                min_years = (exp_requirements.get('minimum_years', 'Not specified') 
                            if isinstance(exp_requirements, dict) 
                            else 'Not specified')
                
                exp_score = self.calculate_experience_match(
                    min_years,
                    resume.get('experience', [])
                )
                
                # Handle education requirements safely
                edu_requirements = job.get('educationRequirements', {})
                edu_score = self.calculate_education_match(
                    resume.get('education', []),
                    edu_requirements if isinstance(edu_requirements, dict) else {}
                )
                
                project_score = self.calculate_project_match(
                    resume.get('projects', []),
                    job_title,
                    required_skills
                )
                
                # Calculate weighted total score using DB weights
                total_score = (
                    (skills_score * weights['skills_weight']) +
                    (exp_score * weights['experience_weight']) +
                    (edu_score * weights['education_weight']) +
                    (project_score * weights['project_weight'])
                )
                
                matches.append({
                    'candidate_name': resume_name,
                    'job_title': job_title,
                    'job_type': self.predict_job_type(job_title),
                    'skills_match': skills_score,
                    'experience_match': exp_score,
                    'education_match': edu_score,
                    'project_match': project_score,
                    'total_score': total_score,
                    'weights_used': weights,
                    'required_skills': required_skills,
                    'candidate_skills': resume.get('technicalSkills', [])
                })
        
        return sorted(matches, key=lambda x: x['total_score'], reverse=True)


def main():
    # API and Hugging Face tokens
    HUGGINGFACE_TOKEN = "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"
    API_TOKEN = "napi_0mgqs6es15ugo29dv8n0ql3exbtuaksx21p752zd8odxovsy26p8pmrxnhbqw6o0"
    
    # Initialize matching system
    matcher = MatchingSystem(HUGGINGFACE_TOKEN, API_TOKEN)
    
    # Calculate and display matches
    matches = matcher.calculate_matches()
    
    print("\n=== Match Results ===")
    for match in matches:
        print(f"\nCandidate: {match['candidate_name']}")
        print(f"Job: {match['job_title']} ({match['job_type']})")
        print(f"Required Skills: {', '.join(match['required_skills'])}")
        print(f"Candidate Skills: {', '.join(match['candidate_skills'])}")
        print("\nMatch Scores:")
        print(f"Skills Match: {match['skills_match']:.2f}")
        print(f"Experience Match: {match['experience_match']:.2f}")
        print(f"Education Match: {match['education_match']:.2f}")
        print(f"Project Match: {match['project_match']:.2f}")
        print(f"Total Score: {match['total_score']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
I'm Manav — a Data Scientist and ML/MLOps/AI/GenAI Engineer. I specialize in building scalable, intelligent systems that connect data to business goals. With experience in delivering real-world AI and ML solutions, I’m confident in my ability to directly contribute to the responsibilities and vision outlined in this role.

Your job is to write a cold email to the client regarding the job mentioned above, describing Manav's capabilities in fulfilling their needs.  
Also highlight how his background in Data Science, ML, AI, GenAI, and MLOps directly applies to the role.  
Use a confident but conversational tone. Dont always write Data Scientist, use synonyms like ML Engineer, AI Engineer, GenAI Engineer, MLOps Engineer, etc. as per the job description.
Do not provide a preamble. 
This is my resume details 
use if needed:
Manav Desai Email : manavdesai170303@gmail.com
GitHub: https://github.com/Manav54321 Mobile : +91 8320936239

Skills
• Mathematics: Linear Algebra, Calculus and Optimization, Statistics & Probability
• Languages: Python, SQL
• Tech Stack: Machine Learning, Deep Learning, Feature Engineering, EDA, Supervised, Unsupervised, Reinforcement
Learning, NLP
• Development & Tools: GitHub, Git, Jupyter Notebook, conda, Google Colab, Numpy, Pandas, Matplotlib, Seaborn,
Scikit-learn
• Frameworks: TensorFlow, PyTorch, Streamlit, Keras, Transformers, FastAPI, Flask
• MLOps & Cloud: MLflow (Experiment Tracking, Model Registry, Model Deployment), CI/CD Pipelines, GitHub
Actions, AWS (S3, EC2, IAM, ECR), Google Cloud Workspace, Docker, DockerHub, MongoDB, DVC, YAML, DagsHub,
Kubernetes, Prometheus, & Grafana
Experience
•
NJ Wealth Surat, India
Machine Learning Engineer Jan - Apr 2025
Projects
• Capstone [MLOps] (Repo.)- Development Phase
DVC, MLflow, AWS(S3, EC2, IAM, ECR, EKS), FastAPI, Docker, GitHub Actions(CI/CD), Prometheus, Grafana Apr 2025
◦ End-to-End System Design: Built a fully automated, production-ready MLOps pipeline for movie reviews
sentiment analysis. Structured the repo using cookiecutter and implemented data ingestion, preprocessing, feature
engineering, model training, evaluation, and registration with MLflow on DagsHub. Integrated DVC for data versioning
and AWS S3 for remote storage. Developed a FastAPI-based REST service, containerized with Docker, and deployed on
AWS EC2 & EKS. CI/CD pipeline established using GitHub Actions to automate testing, retraining, and model/image
deployment to AWS ECR. Enabled real-time monitoring with Prometheus & Grafana on EC2.
• Vehicle Insurance Data Pipeline [MLOps] (Repo.)
Python, MongoDB, Scikit-learn, AWS (S3, EC2, IAM, ECR), FastAPI, Docker, GitHub Actions(CI/CD) Mar 2025
◦ End-to-End System Design: Designed and built a production-grade MLOps pipeline for vehicle insurance risk
assessment, integrating MongoDB Atlas, AWS S3, Scikit-learn, FastAPI, Docker, AWS EC2, GitHub Actions,
and AWS ECR. Implemented automated data processing, model training, deployment, CI/CD, real-time
monitoring, and versioning. Developed a FastAPI-based prediction service, containerized with Docker, and
deployed on AWS EC2. Established self-hosted runners and automated retraining via GitHub Actions & AWS
ECR, ensuring scalability, and performance tracking.
Cost Estimation Network (Repo.)
•
Python, NumPy, Matplotlib, Seaborn, Pandas, Scikit-learn, Streamlit, Git & GitHub, Google Colab July 2024
◦ Engineering Breakdown: Developed a machine learning model for laptop price prediction, improving R² from 0.74
to 0.88 using ensemble methods. Processed and cleaned data, performed EDA, and applied feature engineering to
enhance learning. Optimized multiple supervised learning models and deployed a Streamlit app for real-time
predictions.
• AgroNexus (Repo.)
Python, Jupyter Notebook, NumPy, Matplotlib, Pandas, Scikit-learn, Streamlit, Git & GitHub May 2024
◦ Engineering Breakdown: Developed a machine learning platform for crop selection, pricing, fertilizer
recommendation, and yield prediction. Processed agricultural data, applied dimensionality reduction and feature
engineering, and optimized supervised learning models. Deployed a Streamlit-based interactive tool, providing
farmers with real-time insights and personalized recommendations. 
### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
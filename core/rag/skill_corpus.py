from typing import Dict, Any
import time

from core.rag.vectorstore import add_document, SKILLS_INDEX_NAME 

DOMAIN_CANONICAL_SKILLS: Dict[str, Dict[str, Dict[str, Any]]] = {
    
    # 1. TECHNOLOGY DOMAIN
    "TECH": {
        
        # A. DATA ANALYST SUB-DOMAIN (Focus on reporting, metrics, basic data cleanup)
        "DATA_ANALYST": {
            "CAN_ANALYTICS_TOOLS": {
                "canonical_name": "Business Intelligence and Reporting Tools",
                "description": "Expertise in data visualization, dashboard creation, and using BI platforms.",
                "keywords": ["Tableau", "Power BI", "Looker", "Excel", "Spreadsheets", "Data Visualization", "Reporting", "Metrics"],
                "weight": 0.9  # High importance for analysts
            },
            "CAN_DB_SQL_DA": {
                "canonical_name": "SQL for Data Retrieval and Analysis",
                "description": "Strong skills in writing complex SQL queries for data extraction and preliminary analysis.",
                "keywords": ["SQL", "PostgreSQL", "MySQL", "Data Extraction", "Query Optimization", "Data Retrieval"],
                "weight": 0.8
            },
            "CAN_PYTHON_SCRIPTING": {
                "canonical_name": "Basic Python/R for Data Manipulation",
                "description": "Familiarity with Python or R for data cleaning and manipulation tasks.",
                "keywords": ["Python", "R", "Pandas", "data manipulation", "scripting", "data cleaning"],
                "weight": 0.7
            },
        },

        # B. ML ENGINEER SUB-DOMAIN (Focus on production, MLOps, deployment)
        "ML_ENGINEER": {
            "CAN_CLOUD_M_PROD": {
                "canonical_name": "MLOps and Production Deployment",
                "description": "Experience building end-to-end ML pipelines, model serving, monitoring, and MLOps tools.",
                "keywords": ["MLOps", "Kubeflow", "Airflow", "MLFlow", "Model Deployment", "Production ML", "CI/CD for ML"],
                "weight": 1.0 # Mission-critical for MLOps roles
            },
            "CAN_K8S_ORCHESTRATION": {
                "canonical_name": "Container and Kubernetes Orchestration",
                "description": "Deep knowledge of Docker and Kubernetes for microservices and scalable ML model serving.",
                "keywords": ["Kubernetes", "K8s", "Docker", "Containers", "DevOps orchestration", "Helm"],
                "weight": 0.95 
            },
            "CAN_PYTHON_ENG": {
                "canonical_name": "Production-Ready Python and Software Engineering",
                "description": "Expertise in writing clean, tested, and high-performance Python code for systems integration.",
                "keywords": ["Python", "Software Engineering", "Unit Testing", "TDD", "Clean Code", "Microservices (Python)"],
                "weight": 0.9
            },
        },
        
        # C. AI ENGINEER SUB-DOMAIN (Focus on Generative AI, LLMs, NLP)
        "AI_ENGINEER": {
            "CAN_NLP_LLMS": {
                "canonical_name": "Natural Language Processing and Large Language Models",
                "description": "Expertise in NLP techniques, Transformer models, prompt engineering, and LLM fine-tuning.",
                "keywords": ["NLP", "LLMs", "Generative AI", "Transformers", "BERT", "GPT", "Prompt Engineering", "Fine-tuning"],
                "weight": 1.0 # Mission-critical for AI roles
            },
            "CAN_RAG_VECTORS": {
                "canonical_name": "RAG and Vector Databases",
                "description": "Experience building Retrieval-Augmented Generation (RAG) systems using vector databases.",
                "keywords": ["RAG", "Vector Database", "Pinecone", "Milvus", "Vectorization", "LangChain", "LlamaIndex"],
                "weight": 0.95
            },
            "CAN_DATA_SCALING": {
                "canonical_name": "Data Scaling and Distributed Computing",
                "description": "Proficiency in tools and concepts for processing large-scale data for model training.",
                "keywords": ["Spark", "PySpark", "Hadoop", "Distributed Computing", "Big Data"],
                "weight": 0.8
            },
        },
        
        # D. SOFTWARE ENGINEER SUB-DOMAIN (Refined from previous structure)
        "SOFTWARE_ENGINEER": {
            "CAN_JAVA_BACKEND": {
                "canonical_name": "Java Spring Boot Backend Development",
                "description": "Proficiency in modern Java and the Spring Boot framework for building robust, scalable backend services.",
                "keywords": ["Java", "Spring Boot", "REST API", "Microservices", "OOP", "Backend", "Spring Cloud"],
                "weight": 0.9
            },
            "CAN_FRONTEND_REACT": {
                "canonical_name": "Modern React Frontend Development",
                "description": "Proficiency in modern JavaScript and React frameworks for building user interfaces.",
                "keywords": ["React", "Redux", "Hooks", "JavaScript", "TypeScript", "Frontend", "UI/UX", "Next.js"],
                "weight": 0.8
            },
            "CAN_CICD_DEVOPS": {
                "canonical_name": "CI/CD and Testing",
                "description": "Experience with continuous integration/continuous deployment pipelines and automated testing practices.",
                "keywords": ["CI/CD", "Jenkins", "GitLab CI", "GitHub Actions", "Unit Testing", "TDD", "Integration Testing"],
                "weight": 0.7
            },
        }
    },
    
    # 2. HUMAN RESOURCES (HR) DOMAIN
    "HR": {
        # E. GENERAL HR SUB-DOMAIN (Kept for simplicity in this example)
        "GENERAL_HR": {
            "CAN_HR_FMLA_COMPLIANCE": {
                "canonical_name": "FMLA/EEO/FLSA Compliance",
                "description": "Expertise in federal and state labor laws, including FMLA, Equal Employment Opportunity, and Fair Labor Standards Act.",
                "keywords": ["FMLA", "EEO", "FLSA", "Compliance", "Labor Law", "HR regulations"],
                "weight": 1.0 
            },
            "CAN_HR_RECRUITMENT_LIFECYCLE": {
                "canonical_name": "Full-Cycle Recruitment Management",
                "description": "Ability to manage the entire hiring process from sourcing to offer letter and onboarding.",
                "keywords": ["Recruitment", "Sourcing", "Applicant Tracking Systems", "ATS", "Onboarding", "Talent Acquisition"],
                "weight": 0.8
            },
            "CAN_HR_BENEFIT_ADMIN": {
                "canonical_name": "Benefits and Compensation Administration",
                "description": "Experience managing employee benefits packages, compensation strategies, and payroll coordination.",
                "keywords": ["Benefit Administration", "Compensation", "Payroll", "401k", "Health Insurance", "Total Rewards"],
                "weight": 0.7
            },
        }
    }
}


# --- Pipeline Function ---
def build_skill_corpus():
    """
    Builds the Skill Corpus index by iterating over domain and sub-domain definitions,
    and indexing each skill with appropriate metadata tags for filtering.
    """
    
    print(f"Starting to build Skill Corpus into index: '{SKILLS_INDEX_NAME}'...")
    total_skills_indexed = 0
    
    # Check total skills to index
    for domain, sub_domains in DOMAIN_CANONICAL_SKILLS.items():
        for sub_domain, skills_dict in sub_domains.items():
            total_skills_indexed += len(skills_dict)
            
    print(f"Total skills to index across all domains: {total_skills_indexed}")
    indexed_count = 0
    
    for domain, sub_domains in DOMAIN_CANONICAL_SKILLS.items():
        print(f"\n--- Indexing Domain: {domain} ---")
        
        for sub_domain, skills_dict in sub_domains.items():
            print(f"  --- Indexing Sub-Domain: {sub_domain} ({len(skills_dict)} skills) ---")
            
            for skill_id, skill_data in skills_dict.items():
                
                canonical_name = skill_data['canonical_name']
                description = skill_data['description']
                keywords = ", ".join(skill_data['keywords'])
                weight = skill_data['weight']

                # 1. CREATE AUGMENTED CHUNK for the Skill
                vector_content = f"""
                CANONICAL SKILL ID: {skill_id}
                DOMAIN: {domain}
                SUB-DOMAIN: {sub_domain}
                NAME: {canonical_name}
                DESCRIPTION: {description}
                RELATED TERMS/SYNONYMS: {keywords}
                """
                
                # 2. METADATA (CRITICAL: Store both domain and sub_domain for filtering)
                metadata = {
                    "canonical_name": canonical_name,
                    "weight": weight,
                    "skill_id": skill_id,
                    "domain": domain,
                    "sub_domain": sub_domain 
                }

                # 3. EMBED & STORE
                try:
                    add_document(
                        id=skill_id, 
                        content=vector_content.strip(), 
                        metadata=metadata,
                        index_name=SKILLS_INDEX_NAME
                    )
                    
                    indexed_count += 1
                    print(f"[{indexed_count}/{total_skills_indexed}] SUCCESS: {domain}/{sub_domain} Skill: {canonical_name} (Weight: {weight})")
                    time.sleep(0.5) 
                    
                except Exception as e:
                    print(f"[ERROR] indexing skill {skill_id}: {e}")

    print(f"\n--- Skill Corpus Build Complete. Total Skills Indexed: {indexed_count} ---")

if __name__ == "__main__":
    build_skill_corpus()
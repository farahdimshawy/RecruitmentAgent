from core.rag.retrieval import rag_evaluate_resume

print("\n--- TESTING FULL RAG EVALUATION PIPELINE ---")

resume_skills = ["python", "sql"]
job_description = "We need a data analyst with python, sql, machine learning, agile skills."

result = rag_evaluate_resume(resume_skills, job_description)

print("\nExpanded Resume Skills:")
print(result["expanded_resume_skills"])

print("\nExpanded Job Skills:")
print(result["expanded_job_skills"])

print("\nFinal Similarity Score:")
print(result["similarity_score"])

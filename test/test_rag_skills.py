from core.rag.retrieval import expand_skills

print("\n--- TESTING SKILL EXPANSION VIA RAG ---")

skills = ["python", "sql"]
expanded = expand_skills(skills, k=2)

print(expanded)

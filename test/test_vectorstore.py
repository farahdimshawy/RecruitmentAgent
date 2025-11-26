from core.rag.vectorstore import (
    add_documents,
    retrieve_context,
    get_vectorstore,
    clear_index
)
import pinecone
print("\n--- TESTING VECTORSTORE ---")

# 1. Clear index so the test is clean
try:
    clear_index()
    print("Index cleared.")
except pinecone.exceptions.NotFoundException:
    print("Namespace not found, nothing to delete.")

# 2. Add sample documents
docs = [
    "Python is a programming language used for data science.",
    "SQL is used for data analysis and working with databases.",
    "Machine learning includes models like Random Forest and XGBoost.",
    "Project management uses tools like Jira and Agile methodologies."
]

add_documents(docs)
print("Documents added successfully!")

# 3. Retrieve similar content
query = "data science with python"
results = retrieve_context(query, k=3)

print("\nTop Results for:", query)
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content[:80]}...")

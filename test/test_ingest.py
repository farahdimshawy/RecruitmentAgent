# test/test_ingest.py
from core.rag.vectorstore import (
    add_documents,
    retrieve_context,
    get_vectorstore,
    clear_index
)
import time
from itertools import islice
import pandas as pd
import pinecone

print("\n--- TESTING VECTORSTORE WITH CSV RESUMES ---")

# 1. Clear index so the test is clean
try:
    clear_index()
    print("Index cleared.")
except pinecone.exceptions.NotFoundException:
    print("Namespace not found, nothing to delete.")

# 2. Load CSVs and create documents
csv_files = ["/Users/farah/Users/farah/IdeaProjects/RecruitmentAgent/data/rag_corpus/rag_corpus.csv"]
all_docs = []

for file in csv_files:
    print(f"\nLoading {file}...")
    df = pd.read_csv(file)
    
    for idx, row in df.iterrows():
        # Optional: prepend title/category
        text = f"{row['Category']}\n\n{row['text']}" if 'Category' in row else row['text']
        all_docs.append(text)

print(f"\nTotal documents to add: {len(all_docs)}")


BATCH_SIZE = 2
DELAY_SECONDS = 7 

# Using a generator to iterate over the entire document list in chunks
def batch_iterator(iterable, size):
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk

# We'll limit the total documents to 50 for a quick test
doc_limit = 50
documents_to_process = all_docs[:doc_limit]
print(f"Processing {len(documents_to_process)} documents in batches of {BATCH_SIZE}...")

for i, batch in enumerate(batch_iterator(documents_to_process, BATCH_SIZE)):
    print(f"\nAdding batch {i+1} ({len(batch)} documents)...")
    
    # 3. Call the function with the small batch
    add_documents(batch)
    print(f"Batch {i+1} added successfully.")
    
    # 4. Wait to avoid rate limit errors (Crucial step!)
    if (i + 1) * BATCH_SIZE < len(documents_to_process):
        print(f"Waiting for {DELAY_SECONDS} seconds to respect API rate limits...")
        time.sleep(DELAY_SECONDS)

print("\n--- All Document Batches Added Successfully! ---")


# sample_docs = all_docs[:10]  # only first 10 resumes
# add_documents(sample_docs)
# print("Documents added successfully!")

# 4. Test retrieval
query = "data science with python"
results = retrieve_context(query, k=3)

print("\nTop Results for query:", query)
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content[:120]}...")

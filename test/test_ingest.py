# test/test_ingest.py
from core.rag.vectorstore import (
    add_document,
    retrieve_vector_data,
    clear_index,
    DEFAULT_INDEX_NAME
)
import time
from itertools import islice
import pandas as pd
import pinecone
import uuid 

print("\n--- TESTING VECTORSTORE WITH CSV RESUMES ---")


INDEX_TO_USE = DEFAULT_INDEX_NAME
# try:
#     clear_index(INDEX_TO_USE) 
#     print(f"Index '{INDEX_TO_USE}' cleared.")
# except pinecone.exceptions.NotFoundException:
#     print(f"Index '{INDEX_TO_USE}' not found, nothing to delete.")
# except Exception as e:
#     # Handle the case where the index doesn't exist but the client throws a different error
#     print(f"An error occurred while clearing the index: {e}")


# 2. Load CSVs and create documents
csv_files = ["./data/rag_corpus/tech_corpus.csv"]
all_docs = []

for file in csv_files:
    print(f"\nLoading {file}...")
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file}. Skipping.")
        continue
    
    for idx, row in df.iterrows():
        # Create a unique ID for the vector
        doc_id = str(uuid.uuid4())
        
        # Prepare the text content
        text = row['text']
        metadata = {}

        # Optional: prepend title/category if columns exist
        if 'Category' in row:
            text = f"{row['Category']}\n\n{text}" 
            metadata['category'] = row['Category']
        
        # Add metadata for the source file and original index
        metadata['source_file'] = file
        metadata['original_index'] = idx
        
        all_docs.append({'id': doc_id, 'content': text, 'metadata': metadata})

print(f"\nTotal documents to add: {len(all_docs)}")


BATCH_SIZE = 2
DELAY_SECONDS = 7 

doc_limit = 100
documents_to_process = all_docs[:doc_limit]
print(f"Processing {len(documents_to_process)} documents individually...")


for i, doc_data in enumerate(documents_to_process):
    
    # 3. Call the new add_document function for each document
    add_document(
        id=doc_data['id'],
        content=doc_data['content'],
        metadata=doc_data['metadata'],
        index_name=INDEX_TO_USE
    )
    
    if (i + 1) % BATCH_SIZE == 0:
        print(f"\nDocument {i+1} added. Waiting for {DELAY_SECONDS} seconds...")
        
        # 4. Wait to avoid rate limit errors (Crucial step!)
        if (i + 1) < len(documents_to_process):
            time.sleep(DELAY_SECONDS)
    else:
        # Print progress without new line for documents within a 'batch' group
        print(".", end="", flush=True) 

print("\n--- All Document Batches Added Successfully! ---")


# 5. Test retrieval
query = "data science with python"
# Call the new function retrieve_vector_data
results = retrieve_vector_data(query, k=3, index_name=INDEX_TO_USE) 

print("\nTop Results for query:", query)

for i, match in enumerate(results.get("matches", [])):
    
    content = match.get('metadata', {}).get('content', 'Content not found.')
    score = match.get('score', 0.0)
    
    print(f"\n{i+1}. Score: {score:.4f}")
    print(f"   Content: {content[:200]}...")
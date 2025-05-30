 
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np

#openai.api_key = "sk-proj-DawuPHz0KMSm3Fi9vB15xDCIBVtBObKMPBPs8TLRYxDkVEdIQ4z7OejWGsNSB06wdJaXqx8mQhT3BlbkFJCinGuU0b5xM4ZLIpUBtQnGWB7Kb6ltUnULfb2CpFVhT1Z0SsuXCE5L8nS9SkQqzl"
import os
import openai
#openai.api_key = "sk-proj-DawuPHz0KMSm3Fi9vB15xDCIBVtBObKMPBPs8TLRYxDkVEdIQ4z7OejWGsNSB06wdJaXqx8mQhT3BlbkFJCinGuU0b5xM4ZLIpUBtQnGWB7Kb6ltUnULfb2CpFVhT1Z0SsuXCE5L8nS9SkQqzl"



openai.api_key ="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"


# Load fine-tuned model and quotes
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

data = load_dataset("Abirate/english_quotes")
df = data['train'].to_pandas()
df.dropna(subset=['quote', 'author'], inplace=True)
df['text'] = df['quote'] + " - " + df['author'] + " (" + df['tags'].apply(lambda x: ', '.join(x)) + ")"

# Embed and index with FAISS
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Retrieval function
def retrieve_quotes(query, top_k=5):
    query_vec = model.encode([query])
    scores, indices = index.search(query_vec, top_k)
    results = df.iloc[indices[0]]
    return results, scores[0]



# Generator using OpenAI
def generate_answer(query):
    results, scores = retrieve_quotes(query)
    context = "\n".join(results['text'].tolist())
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Use the following quotes to answer the user's query."},
            {"role": "user", "content": f"Quotes:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message['content'], results, scores
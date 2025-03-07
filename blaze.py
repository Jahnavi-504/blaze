from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import torch
from psycopg2 import pool
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template rendering (for a basic search page)
templates = Jinja2Templates(directory="templates")

# Load device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

query_encoder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}
CONN_POOL = pool.ThreadedConnectionPool(1, 32, **DB_PARAMS)

# Partitions list
PARTITIONS = [f"rag_dup_part_{i}" for i in range(1, 17)]


# Request model for search queries
class SearchQuery(BaseModel):
    query: str

# Response model for search results (id is a string and pdf_url is a string)
class SearchResult(BaseModel):
    id: str
    pdf_url: str

def get_db_connection():
    return CONN_POOL.getconn()

def release_db_connection(conn):
    CONN_POOL.putconn(conn)

@app.get("/")
async def root():
    return RedirectResponse(url="/search")

# Serve a search page (if you have an index.html in your templates)
@app.get("/search")
async def search_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Search endpoint: takes a query and returns a list of SearchResult (id and pdf_url)
@app.post("/search", response_model=List[SearchResult])
async def search(query_data: SearchQuery):
    query = query_data.query
    print(f"Received query: {query}")
    results = search_and_rerank_using_medcpt_encoder(query)
    return results

def vector_search_partition(query, qry_embed, partition):
    conn = get_db_connection()
    start_time = time.time()
    try:
        cur = conn.cursor()
        cur.execute("SET max_parallel_workers_per_gather = 0;")
        vec_query = f"""
        SELECT id, pdf_text, keywords, embedding, pdf_url
        FROM {partition}
        ORDER BY embedding <#> %s::vector ASC
        LIMIT 10;
        """
        cur.execute(vec_query, (qry_embed.tolist(),))
        vec_results = cur.fetchall()
        elapsed_time = time.time() - start_time
        print(f"Vector Search Time for {partition}: {elapsed_time:.4f} seconds")
        return [
            {"id": row[0], "pdf_text": row[1], "keywords": row[2], "embedding": row[3], "pdf_url": row[4]}
            for row in vec_results
        ]
    except Exception as e:
        print(f"Error in {partition}: {e}")
        return []
    finally:
        cur.close()
        release_db_connection(conn)

def tsvector_search_partition(query, partition):
    conn = get_db_connection()
    start_time = time.time()
    try:
        cur = conn.cursor()
        cur.execute("SET max_parallel_workers_per_gather = 0;")
        ts_query = f"""
        SELECT id, pdf_text, keywords, pdf_url, ts_rank(keywords_tsvector, plainto_tsquery('english', %s)) AS rank
        FROM {partition}
        WHERE keywords_tsvector @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT 10;
        """
        cur.execute(ts_query, (query, query))
        ts_results = cur.fetchall()
        elapsed_time = time.time() - start_time
        print(f"TSVector Search Time for {partition}: {elapsed_time:.4f} seconds")
        return [
            {"id": row[0], "pdf_text": row[1], "keywords": row[2], "pdf_url": row[3]}
            for row in ts_results
        ]
    except Exception as e:
        print(f"Error in {partition}: {e}")
        return []
    finally:
        cur.close()
        release_db_connection(conn)

def search_and_rerank_using_medcpt_encoder(query):
    """Perform vector search, TSVECTOR search, merge results, and re-rank using MedCPT."""
    print(f"Starting Parallel Search for Query: {query}")
    start_time = time.time()
    with torch.no_grad():
        query_token = query_tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        query_embed = query_encoder(**query_token).last_hidden_state[:, 0, :].cpu().numpy()[0]
    all_vec_docs, all_ts_docs = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        vec_futures = {
            executor.submit(vector_search_partition, query, query_embed, partition): partition
            for partition in PARTITIONS
        }
        ts_futures = {
            executor.submit(tsvector_search_partition, query, partition): partition
            for partition in PARTITIONS
        }
        for future in concurrent.futures.as_completed(vec_futures):
            try:
                result = future.result()
                all_vec_docs.extend(result)
            except Exception as e:
                print(f"Vector Search Error: {e}")
        for future in concurrent.futures.as_completed(ts_futures):
            try:
                result = future.result()
                all_ts_docs.extend(result)
            except Exception as e:
                print(f"TSVector Search Error: {e}")
    print(f"Parallel Search Completed in {time.time() - start_time:.4f} seconds")
    print(f"Vector Search Retrieved: {len(all_vec_docs)} docs")
    print(f"TSVector Search Retrieved: {len(all_ts_docs)} docs")
    # Merge and process results
    combined_results = all_vec_docs + all_ts_docs  # Merge lists
    search_results = []
    for item in combined_results:
        item["id"] = str(item["id"])

    for doc in combined_results:
        search_results.append({
            "id": doc["id"],
            "pdf_url": doc["pdf_url"]
        })
    
    print(f"Total Results: {len(search_results)}")

    return search_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, timeout_keep_alive=300)
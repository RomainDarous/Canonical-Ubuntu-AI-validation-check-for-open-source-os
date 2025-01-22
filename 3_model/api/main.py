import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import random
import pandas as pd
import io
from fastapi import UploadFile, File
from bs4 import BeautifulSoup

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
def cosim(vec1, vec2) -> float :
    vec1 = torch.tensor(vec1)
    vec2 = torch.tensor(vec2)
    dot_product = torch.dot(vec1, vec2)  # Efficient dot product
    norm_vec1 = torch.linalg.norm(vec1)  # Norm of vec1
    norm_vec2 = torch.linalg.norm(vec2)  # Norm of vec2
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity.item()

def remove_html(sentence: str) -> str:
    """Removes html content from the data.

    Args:
        sentence (str): The string to process.

    Returns:
        str: The string free from any html code.
    """
    if isinstance(sentence, str): 
        return BeautifulSoup(sentence, 'html.parser').get_text()
    return sentence

def cleaning(df: pd.DataFrame) -> pd.DataFrame:
        """Performs cleaning of the dataset in input for training.

        Args:
            df (pd.Dataframe): the input dataframe to clean
        Returns:
            pd.DataFrame: a cleaned version of the input dataframe
        """
        type_clean = [
            r'@\w+',                                # Remove @"content "
            r'\n+|\t+|\\n+',
            r'http\S+|www\S+',                      # Remove URLs
            r'\[UTF-[^\]]+\]',                      # Remove UTF characters
            r'(\()*(%[^)]+)(\))*',                  # Remove (%s) characters
            r'^-+|-',                               # Replace leading/trailing dashes
            r'(?<!^)(?=[A-Z])(?![A-Z])',            # Split attached words
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',         # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b', # IPv6
            r'\b(?:\d+\.)+\d+\b',                    # Removing 4.6.5.3-like sequences
        ]

        char_clean = [
            r'(\$)',                                    # Remove ($){} characters
            r'\(\((\()*((\s)*)(\))* | (\()*((\s)*)(\))*\)\) | \s\(\s | \s\)\s | \((\()*((\s)*)(\))*\)',   # Remove ((())))) types of strings
            r'\{\{(\{)*((\s)*)(\})* | (\{)*((\s)*)(\})*\}\} | \s\{\s | \s\}\s | \{(\{)*((\s)*)(\})*\}',   # Same for {
            r'\[\[(\[)*((\s)*)(\])* | (\[)*((\s)*)(\])*\]\] | \s\[\s | \s\]\s | \[(\[)*((\s)*)(\])*\]',   # Same for {
            r'\/(\/)*',                                 # Remove all /(/)*
            r'(?<=\s)([^\w\s?!:;]+)(?=\s)|^([^\w\s?!:;]+)(?=\s)|(?<=\s)([^\w\s?!:;]+)$',  # Remove special characters
            r'``*| \'(\')*|""*|””*|““*',                # Remove various quotation marks
        ]

        for column in df.columns:         
            # Cleaning all the columns
            try : 
                df.loc[:,column] = df.loc[:,column].apply(remove_html)
                df.loc[:,column] = df.loc[:,column].str.replace('|'.join(type_clean), ' ', regex=True)
                df.loc[:,column] = df.loc[:,column].str.replace('|'.join(char_clean), ' ', regex=True)

                # Removing excessive spaces
                df.loc[:,column] = df.loc[:,column].str.replace(r'\t(\t)*', ' ', regex=True)
                df.loc[:,column] = df.loc[:,column].str.replace(r'\s+', ' ', regex=True).str.strip()
            except Exception as e :
                print(f"Error on column {column} : {e}")
                print(df.head())
                continue
        
        # Removing duplicates
        df = (df.drop_duplicates(inplace=False)).loc[:, :]

        if 'en' in df.columns :
            return df[df['en'].str.split(' ').str.len() <= 128]
        else :
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
            return df

# Pydantic model for request validation
class SimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str

# Pydantic model for response
class SimilarityResponse(BaseModel):
    similarity_score: float

# Pydantic model for file processing response
class FileProcessingResponse(BaseModel):
    results: List[dict]

# Global variables
model = None
workers = []
fifo_queue = asyncio.Queue()

async def worker():
    while True:
        try:
            print(f"Waiting for job... (queue size: {fifo_queue.qsize()})")
            job = await fifo_queue.get()
            result = await job()
            print(f"Job completed successfully")
        except Exception as e:
            print(f"Error processing job: {str(e)}")
        finally:
            fifo_queue.task_done()

async def calculate_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the cosine similarity between two sentences using the loaded model.
    """
    if model is None:
        raise RuntimeError("Model not initialized")
    
    try:
        # Encode the sentences      
        vec1 = model.encode(sentence1)
        vec2 = model.encode(sentence2)
        return cosim(vec1, vec2)
    
    except Exception as e:
        raise RuntimeError(f"Error calculating similarity: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model and workers
    global model, workers
    print("Application is starting up...")
    
    try:
        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the existing SentenceTransformer model
        #model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device=device)
        model = SentenceTransformer('RomainDarous/one_epoch_dot_product_mistranslation_model', device=device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        
        # Initialize workers
        num_workers = 3  # Adjust based on your needs
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        print(f"Created {num_workers} worker tasks")
        
        yield
        
    finally:
        # Shutdown: Cleanup workers and model
        print("Application is shutting down...")
        
        # Cancel all worker tasks
        for worker_task in workers:
            worker_task.cancel()
        
        # Wait for all tasks to complete
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        
        # Clear model
        model = None
        workers = []

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(request: SimilarityRequest):
    """
    Endpoint to calculate similarity between two sentences.
    """
    async def process_similarity():
        try:
            score = await calculate_similarity(request.sentence1, request.sentence2)
            return {"similarity_score": score}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    await fifo_queue.put(process_similarity)
    return await process_similarity()


@app.post("/process-file", response_model=FileProcessingResponse)
async def process_file(file: UploadFile):
    if file.filename is None or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read the CSV file
        contents = await file.read()
        print(type(contents), file)
        df = pd.read_csv(io.BytesIO(contents), sep=',', encoding='utf-8')
        print("reached that part", df)
        df = cleaning(df.loc[:,:])

        col1=""
        col2=""

        # Validate columns
        if not len(list(df.columns)) == 2:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain two columns"
            )
        else :
            col1 = df.columns[0]
            col2 = df.columns[1]
        
        results = []
        
        # Process each row
        for _, row in df.iterrows():
            score = await calculate_similarity(row[col1], row[col2])
            results.append({
                'sentence1': row[col1],
                'sentence2': row[col2],
                'score': score
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
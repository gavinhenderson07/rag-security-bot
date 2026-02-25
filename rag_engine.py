#Gavin's RAG Engine Personal Project
#December, 2025

import os
# Force TF Hub to save the model in a permanent folder inside your project
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.getcwd(), "model_cache")

import re
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


#open file and read the content
def load_text(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        content = file.read()
    return content

def clean_text(content):
    
    #function to clean the input text by removing unwanted special characters and other noise.
    
    content = re.sub(r'#+ ', '', content)

    content = re.sub(r'\n', ' ', content)

    return content


def chunk_text(content, chunk_size=200, overlap=50):
    #function to split the clean text into smaller chunks with overlap to maintain context

    word_list = content.split()
    chunks = []

    #loop through word list and chunk it
    #start at beginning and move by step size for overlap
    start = 0
    stop = len(word_list)
    step = chunk_size - overlap

    #grab words from start to chunk size
    #combine into a single string and add to list of chunks
    for i in range(start, stop, step):
        words = word_list[i : i + chunk_size]
        chunk = ' '.join(words)
        chunks.append(chunk)
    return chunks



#load in and create the model from the url 
def load_embedding_model():
    url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    print("Loading embedding model...")
    model = hub.load(url)
    print("Model loaded successfully!")
    return model


#convert chunks from strings into vectors
#pass strings into model to turn into vectors
def convert_chunks(text_chunks, model):
    vectors = model(text_chunks)

    return vectors


#create the dataframe to hold chunks and vectors with pandas
def create_dataframe(text_chunks, vectors):
    data = {
        "text_chunks": text_chunks,
        "vectors": list(vectors.numpy())
    }

    df = pd.DataFrame(data)
    return df


#using the model, vectors, chunks, and user question
#find the cosine similarity (ie closest related vector) using numpy
#to closely relate the question to the answer text
def search_index(dataframe, query, model, top_k=3):
    #pass in the query, df, and model to find closest related chunk
    query_vector = model([query])
    
    #create a matrix from the vectors in the dataframe
    matrix = np.stack(dataframe['vectors'].values)

    #find cosine similarity using np.inner() and grab the first (only) row
    answer_similarities = np.inner(query_vector, matrix)[0]

    #find the top k similar chunks for more context retention
    top_k_indices = np.argsort(answer_similarities)[-top_k:][::-1]

    #grab the top chunks and their similarities for the bot
    best_scores = answer_similarities[top_k_indices]
    best_chunks = dataframe.iloc[top_k_indices]['text_chunks'].to_list()

    #join the list of top chunks to use for context
    combined_best = "\n\n --New Chunk-- \n\n".join(best_chunks)


    return combined_best, best_scores[0]


def get_or_create_db(model, file_path="knowledge.txt", db_path="vector_db.pkl"):
    #check if the database file already exists
    if os.path.exists(db_path):
        print("Loading saved vector database from disk...")
        #instantly load the pre-computed DataFrame
        return pd.read_pickle(db_path)
        
    else:
        print("Building vector database from scratch...")
        #run your original pipeline
        raw_text = load_text(file_path)
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)
        vectors = convert_chunks(chunks, model)
        df = create_dataframe(chunks, vectors)
        
        #save the DataFrame to your hard drive
        df.to_pickle(db_path)
        print("Database saved locally!")
        
        return df


#use OpenAI api to generate final answer based on best chunk found
#set up OpenAI Clien and load env variables
load_dotenv()
client = OpenAI()

def generate_answer(query, best_chunk):
    #set the message to feed the model
    #include system prompt and user prompt with context
    messages = [
        {"role": "system", "content": "You are a helpful Cybersecurity assistant that provides accurate and concise answers based on the provided context."},
        {"role": "user", "content": f"Context: {best_chunk:} Question: {query}"}
    ]

    #call the openai chat to generate the answer
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = 0.5
    )

    #extract the first answer from the model's response
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    print("--- Starting Test Run ---")
    
    # 1. Load the raw data
    raw_text = load_text("knowledge.txt")
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    
    # 2. Build the 'Brain'
    model = load_embedding_model()
    vectors = convert_chunks(chunks, model)
    df = create_dataframe(chunks, vectors)
    
    # 3. Test a question
    test_query = "How do I remove ransomware?"
    best_chunk, similarity = search_index(df, test_query, model)
    
    # 4. See the result
    final_answer = generate_answer(test_query, best_chunk)
    print(f"AI Answer: {final_answer}")
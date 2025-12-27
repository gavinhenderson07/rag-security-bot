#Gavin's RAG Engine Personal Project
#December, 2025

import re
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np


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

#load and clean data for further processing
raw_data = load_text('knowledge.txt')
clean_data = clean_text(raw_data)


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

text_chunks = chunk_text(clean_data)


#load in and create the model from the url 
def load_embedding_model():
    url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    print("Loading embedding model...")
    model = hub.load(url)
    print("Model loaded successfully!")
    return model

model = load_embedding_model()

#convert chunks from strings into vectors
#pass strings into model to turn into vectors
def convert_chunks(text_chunks, model):
    vectors = model(text_chunks)

    return vectors

vectors = convert_chunks(text_chunks, model)

#using the model, vectors, chunks, and user question
#find the cosine similarity (ie closest related vector) using numpy
#to closely relate the question to the answer text
def search_index(vectors, model, text_chunks, query):
    #model expects an input of a list
    query_vector = model([query])

    #use np to calculate inner product (dot product) with
    #query vector against all chunk vectors
    #add [0] to get the scores as a 1D array, which argmax() needs
    scores = np.inner(query_vector, vectors)[0]

    #find the index with the vector with the best match
    answer_vector = np.argmax(scores)

    return text_chunks[answer_vector], scores[answer_vector]

test_query = "How do I deal with and dispose of ransomeware?"
result, score = search_index(vectors, model, text_chunks, test_query)

print(f"Query: {test_query}")
print(f"Best Match Score: {score}")
print(f"Retrieved Chunk: {result}")
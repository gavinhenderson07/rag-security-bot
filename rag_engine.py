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

#create the dataframe to hold chunks and vectors with pandas
def create_dataframe(text_chunks, vectors):
    data = {
        "text_chunks": text_chunks,
        "vectors": list(vectors.numpy())
    }

    df = pd.DataFrame(data)
    return df

df = create_dataframe(text_chunks, vectors)

#using the model, vectors, chunks, and user question
#find the cosine similarity (ie closest related vector) using numpy
#to closely relate the question to the answer text
def search_index(dataframe, query, model):
    #pass in the query, df, and model to find closest related chunk
    query_vector = model([query])
    
    #create a matrix from the vectors in the dataframe
    matrix = np.stack(dataframe['vectors'].values)

    #find cosine similarity using np.inner() and grab the first (only) row
    answer_similarities = np.inner(query_vector, matrix)[0]

    #find the index of the highest similarity score
    best_match_idx = np.argmax(answer_similarities)

    #fetch the corresponding row from the dataframe
    result_row = dataframe.iloc[best_match_idx]

    return result_row['text_chunks'], answer_similarities[best_match_idx]

test_query = "How do I remove ransomeware on my computer?"
best_chunk, similarity = search_index(df, test_query, model)
print(f"Example Query: {test_query}.")
print(f"Chunk found: {best_chunk}")
print(f"Similarity: {similarity}")


import re


def load_text(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        content = file.read()
    return content

def clean_text(content):
    
    #function to clean the input text by removing unwanted special characters and other noise.
    
    content = re.sub(r'#+ ', '', content)

    content = re.sub(r'\n', ' ', content)

    return content

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

    for i in range(start, stop, step):
        words = word_list[i : i + chunk_size]
        chunk = ' '.join(words)
        chunks.append(chunk)
    return chunks

text_chunks = chunk_text(clean_data)
print(len(text_chunks), "chunks created.")
print(f"Sample Chunk: {text_chunks[1]}")
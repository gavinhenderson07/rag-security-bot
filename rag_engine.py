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

print(clean_data[:500])

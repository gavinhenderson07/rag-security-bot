import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import rag_engine

#url we need for the api
endpoint_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def fetch_cve_data():

    #get the currnt date and yesterday so we can get most recent cves
    now = datetime.now()
    yesterday = now - timedelta(days = 1)

    #set the strict format that the API wants
    date_format = "%Y-%m-%dT%H:%M:%S.000"


    #pass in the parameters to the api to get the data we want
    params = {
        "pubStartDate": yesterday.strftime(date_format),
        "pubEndDate": now.strftime(date_format),
        "resultsPerPage": 5
    }

    print(f"Fetching data from: {endpoint_url}")

    #try/catch block to handle any errors during the API request
    try:
        response = requests.get(endpoint_url, params=params)
        response.raise_for_status() #check for http errors
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
    #parse the data to get the information we need for the chatbot
def parse_cve_data(cve_data):
    parsed_data = []
    for item in cve_data.get("vulnerabilities", []):
        cve_id = item.get("cve", {}).get("id", "N/A")
        description = item.get("cve", {}).get("descriptions", [{}])[0].get("value", "No description available.")
        parsed_data.append({
            "cve_id": cve_id,
            "description": description
        })
    return parsed_data

def update_knowledge_base(parsed_data, file_path="vector_db.pkl"):
    #create a set to hold existing cve ids to prevent duplicate data
    existing_cves = set()

    #try to load the existing df and extract the cve ids we already have
    try:
        old_df = pd.read_pickle(file_path)
        for text in old_df['text_chunks']:
            potential_id = text.split(":")[0].strip()
            
            #only add it to our set if it is actually a cve, ignoring normal playbook text
            if potential_id.startswith("CVE-"):
                existing_cves.add(potential_id)
    except FileNotFoundError:
        print("Knowledge base file not found. Will create a new one.")
        old_df = None

    new_text_chunks = []

    #create the new text chunks from the parsed data, filtering out duplicates
    for item in parsed_data:
        cve_id = item["cve_id"]
        description = item["description"]
        
        #check if the cve is already in our database or invalid before adding
        if cve_id not in existing_cves and cve_id != "N/A":
            new_text_chunks.append(f"{cve_id}: {description}")

    #stop early if there are no new cves to add to save compute time
    if not new_text_chunks:
        print("Database is already up to date. No new CVEs to embed.")
        return

    print(f"Found {len(new_text_chunks)} new CVEs! Embedding now...")

    #load the embedding model
    model = rag_engine.load_embedding_model()

    #create vectors for the new chunks using the model
    new_vectors = rag_engine.convert_chunks(new_text_chunks, model)

    #create the new df by combining the new chunks and vectors
    new_df = rag_engine.create_dataframe(new_text_chunks, new_vectors)

    #combine the old df with the new df, or just use the new df if no old df exists
    if old_df is not None:
        combined_df = pd.concat([old_df, new_df], ignore_index = True)
    else:
        combined_df = new_df
        
    #save the combined data back to disk
    combined_df.to_pickle(file_path)
    print("Knowledge base updated successfully.")
    
    

if __name__ == "__main__":
    raw_data = fetch_cve_data()

    if raw_data:
        parsed_data = parse_cve_data(raw_data)
        update_knowledge_base(parsed_data)
    
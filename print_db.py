import pandas as pd

def inspect_database(file_path="vector_db.pkl"):
    try:
        # Load the database
        df = pd.read_pickle(file_path)
        
        # Print the total number of chunks
        print(f"Total records in database: {len(df)}")
        
        # Print the last 15 text chunks
        print("\n--- Last 15 Records in Database ---")
        
        # .tail(15) grabs the last 15 rows of the DataFrame
        for i, text in enumerate(df['text_chunks'].tail(15), 1):
            print(f"{i}. {text}\n")
            
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure you have run app.py first to generate the database.")

if __name__ == "__main__":
    inspect_database()
import streamlit as st
import rag_engine
import cve_ingest
import pandas as pd


#set page config
st.set_page_config(page_title="Cybersecurity RAG Chatbot", page_icon=":shield:", layout="wide")


#load in model and info at startup, only once (avoiding reloading on every interaction)
@st.cache_resource
def startup():
    #guarantee the foundational NIST playbook is built first
    model = rag_engine.load_embedding_model()
    rag_engine.get_or_create_db(model, "knowledge.txt") 
    
    #check for new daily threats
    print("Checking for daily threat intelligence updates...")
    raw_data = cve_ingest.fetch_cve_data()
    if raw_data:
        parsed_data = cve_ingest.parse_cve_data(raw_data)
        # This safely ignores duplicates and appends new data
        cve_ingest.update_knowledge_base(parsed_data)
        
    #load the final, fully-updated database into memory
    df = pd.read_pickle("vector_db.pkl")
    
    return df, model


#main app function
def main():
    st.title("Cybersecurity RAG Chatbot Assistant :shield:")


    #ADMIN SIDEBAR
    st.sidebar.header("Admin Controls")
    st.sidebar.write("Manually trigger the threat intelligence pipeline.")
    
    #when clicked, this button runs the ingestion script
    if st.sidebar.button("Pull Latest Threat Data"):
        with st.spinner("Fetching latest CVEs from NVD..."):
            
            raw_data = cve_ingest.fetch_cve_data()
            if raw_data:
                parsed_data = cve_ingest.parse_cve_data(raw_data)
                cve_ingest.update_knowledge_base(parsed_data)
                
                # Wipe Streamlit's memory so it knows the database changed
                startup.clear() 
                
                # Refresh the web page
                st.rerun()
            else:
                st.sidebar.error("Failed to reach NVD API.")
    # ---------------------------

    #load in data and model one time
    with st.spinner("Loading knowledge base and embedding model..."):
        df, model = startup()

    #ask for user input
    st.text_input("Enter your cybersecurity question:", key="user_question")

    #button to get answer
    if st.button("Get Advice"):
        user_query = st.session_state.user_question
        #error handling for empty inputs
        if user_query.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            #process question and find answer
            with st.spinner("Searching for the best answer..."):
                best_chunk, similarity = rag_engine.search_index(df, user_query, model)
                answer = rag_engine.generate_answer(user_query, best_chunk)

                st.markdown("### Answer:")
                st.markdown(answer)
            
            #provide expandable section for source context
            with st.expander("Source Context"):
                st.write(f"**Similarity Score:** {similarity}")
                st.info(best_chunk)

if __name__ == "__main__":
    main()
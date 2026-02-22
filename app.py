import streamlit as st
import rag_engine


#set page config
st.set_page_config(page_title="Cybersecurity RAG Chatbot", page_icon=":shield:", layout="wide")


#load in model and info at startup, only once (avoiding reloading on every interaction)
@st.cache_resource
def startup():
    # Load the model first so we can pass it to the database function
    model = rag_engine.load_embedding_model()
    
    # This will load the .pkl file instantly, or build it if it's the first run
    df = rag_engine.get_or_create_db(model, "knowledge.txt")

    return df, model


#main app function
def main():
    st.title("Cybersecurity RAG Chatbot Assistant :shield:")

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
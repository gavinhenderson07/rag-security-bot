# 🛡️ AI Cybersecurity RAG Assistant

A Retrieval-Augmented Generation (RAG) tool designed to assist SOC Analysts. It answers incident response questions based strictly on NIST/AWS Playbooks, preventing LLM hallucinations.

## 🚀 Key Features
* **Semantic Search Engine:** Built from scratch using Python & NumPy to calculate Cosine Similarity between query vectors.
* **Vector Database:** Implemented an in-memory vector store using Pandas and Google's Universal Sentence Encoder (TensorFlow).
* **RAG Pipeline:** Integrated OpenAI's GPT-3.5-Turbo to synthesize human-readable answers from retrieved context chunks.
* **Interactive UI:** Developed a full-stack web interface using Streamlit with caching for optimized model performance.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **AI/ML:** TensorFlow, Universal Sentence Encoder, OpenAI API
* **Data Engineering:** Pandas, NumPy
* **Frontend:** Streamlit

## ⚙️ How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set up your OpenAI API key in a `.env` file.
4.  Run the application: `streamlit run app.py`
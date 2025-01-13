import streamlit as st
import os
import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import dotenv
from using_faiss import extract_text_from_pdf, split_text_into_chunks

# Load environment variables
dotenv.load_dotenv()

# Create FAISS_MODELS directory if it doesn't exist
MODELS_DIR = "FAISS_MODELS"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def get_index_path(chunk_size, chunk_overlap):
    return os.path.join(MODELS_DIR, f"chunk_{chunk_size}_overlap_{chunk_overlap}")

def save_faiss_data(index, chunks, chunk_size, chunk_overlap):
    base_path = get_index_path(chunk_size, chunk_overlap)
    # Save the FAISS index
    faiss.write_index(index, f"{base_path}.index")
    # Save the chunks for later retrieval
    with open(f"{base_path}.chunks", 'wb') as f:
        np.save(f, chunks)

def load_faiss_data(chunk_size, chunk_overlap):
    base_path = get_index_path(chunk_size, chunk_overlap)
    # Load the FAISS index
    index = faiss.read_index(f"{base_path}.index")
    # Load the chunks
    with open(f"{base_path}.chunks", 'rb') as f:
        chunks = np.load(f, allow_pickle=True)
    return index, chunks

def process_document(pdf_path, chunk_size, chunk_overlap):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks with user-defined parameters
    chunks = split_text_into_chunks(pdf_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Generate embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")
    
    # Build FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # Save the index and chunks
    save_faiss_data(index, chunks, chunk_size, chunk_overlap)
    
    return index, chunks

def query_index(query, index, chunks, k=5):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_embedding = np.array(embeddings.embed_query(query)).astype("float32")
    distances, indices = index.search(np.array([query_embedding]), k)
    
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm.predict(prompt)
    
    return response, retrieved_chunks, distances[0]

def main():
    st.title("RAG Parameter Tuning App")
    
    tab1, tab2 = st.tabs(["Create Index", "Query Existing Index"])
    
    with tab1:
        st.header("Create New FAISS Index")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 100, 2000, 500, 50)
            with col2:
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
            
            if st.button("Create Index"):
                with st.spinner("Processing document and creating index..."):
                    try:
                        index, chunks = process_document(temp_path, chunk_size, chunk_overlap)
                        st.success(f"Index created and saved with chunk size {chunk_size} and overlap {chunk_overlap}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    with tab2:
        st.header("Query Existing Index")
        
        # Get list of available indexes
        available_indexes = [f for f in os.listdir(MODELS_DIR) if f.endswith('.index')]
        if not available_indexes:
            st.warning("No indexes available. Please create an index first.")
            return
        
        # Extract parameters from filenames
        index_params = []
        for idx in available_indexes:
            params = idx.replace('.index', '').split('_')
            chunk_size = int(params[1])
            overlap_size = int(params[3])
            index_params.append((chunk_size, overlap_size))
        
        # Let user select parameters
        selected_params = st.selectbox(
            "Select Index Parameters",
            index_params,
            format_func=lambda x: f"Chunk Size: {x[0]}, Overlap: {x[1]}"
        )
        
        # Create a form for the query input
        with st.form(key='query_form'):
            query = st.text_input("Enter your question and press Enter:", key="query_input")
            submit_button = st.form_submit_button('Submit', type='primary')
        
        # Process query only when form is submitted
        if submit_button and query:
            with st.spinner("Processing query..."):
                try:
                    index, chunks = load_faiss_data(selected_params[0], selected_params[1])
                    response, retrieved_chunks, distances = query_index(query, index, chunks)
                    
                    st.subheader("LLM Response")
                    st.write(response)
                    
                    st.subheader("Retrieved Chunks (with distances)")
                    for chunk, distance in zip(retrieved_chunks, distances):
                        with st.expander(f"Chunk (Distance: {distance:.4f})"):
                            st.write(chunk)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
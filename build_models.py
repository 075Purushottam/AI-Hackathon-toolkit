from flask import Flask, request, jsonify
import sys, os
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import re
from rank_bm25 import BM25Okapi
import time
import pickle
app = Flask(__name__)

embeddings = AzureOpenAIEmbeddings(
        model=config.emd_model,
        azure_endpoint = config.emd_endpoint,
        api_key=config.emd_api_key,
        api_version=config.emd_api_version
)

# Directory to save uploaded PDFs and models
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    pdf = fitz.open(pdf_path)
    all_text = []
    for page_num, page in enumerate(pdf):
        text = page.get_text("text")
        if text.strip():
            all_text.append({"page_content": text, "metadata": {"page": page_num + 1}})
    pdf.close()
    return all_text

def split_text(raw_pages, chunk_size=800, chunk_overlap=200):
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    )

    docs = []
    chunk_counter = 0
    for page in raw_pages:
        chunks = splitter.create_documents([page["page_content"]], metadatas=[page["metadata"]])
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_counter
            chunk_counter += 1

        docs.extend(chunks)
        
    return docs

def embed_batch(batch_texts):

    try:
        return embeddings.embed_documents(batch_texts)
    except Exception as e:
        print(f"Error in Batching : {e}")
            
    return None

def batchify(docs,batch_size=75):
    for i in range(0,len(docs),batch_size):
        yield docs[i:i+batch_size]

def build_faiss_index(docs):
        
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    all_vectors = []
    all_texts = []
    all_metas = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        batches = list(batchify(texts, 75))
        meta_batches = list(batchify(metadatas, 75))

        for batch_texts in batches:
            futures.append(executor.submit(embed_batch, batch_texts))

        batch_idx = 0
        for future in as_completed(futures):
            vectors = future.result()
            if vectors:
                # Save embeddings + original text + metadata
                all_vectors.extend(vectors)
                all_texts.extend(batches[batch_idx])
                all_metas.extend(meta_batches[batch_idx])
                print(batch_idx)
            batch_idx += 1


    vectorstore = FAISS.from_embeddings(
        text_embeddings = list(zip(all_texts,all_vectors)),
        embedding=embeddings,
        metadatas=all_metas
    )

    return vectorstore

def build_bm25_index(docs):
    corpus = docs
    tokenized_corpus = [doc.page_content.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

def save_faiss(vectorstore,faiss_path):
    vectorstore.save_local(faiss_path)
    print(f" {faiss_path} Embedding Saved Successfully.")

def save_bm25(bm25,corpus_tokens,bm25_path):
    with open(bm25_path,'wb') as f:
        pickle.dump({'bm25':bm25,'corpus_tokens':corpus_tokens},f)
    print(f' {bm25_path} bm25 Saved Successfully.')

import pandas as pd
import os

def save_model_mapping_to_excel(filename, faiss_path, bm25_path, excel_path="model_mapping.xlsx"):
    """
    Save PDF filename and its FAISS/BM25 model paths into an Excel file.
    
    :param filename: Original PDF filename
    :param faiss_path: Path where FAISS model is saved
    :param bm25_path: Path where BM25 model is saved
    :param excel_path: Path to Excel file (default: model_mapping.xlsx)
    """
    # Check if Excel file exists
    if os.path.exists(excel_path):
        # Load existing data
        df = pd.read_excel(excel_path)
    else:
        # Create new DataFrame
        df = pd.DataFrame(columns=["PDF Filename", "FAISS Path", "BM25 Path"])
    
    # Append new entry
    new_entry = {"PDF Filename": filename, "FAISS Path": faiss_path, "BM25 Path": bm25_path}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save back to Excel
    df.to_excel(excel_path, index=False)
    print(f"Mapping saved to {excel_path}")

@app.route("/build_models", methods=['POST'])
def build_models():
    try:
        # 1. Get PDF file from request
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part in request"}), 400
        
        pdf_file = request.files['file']
        if pdf_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # 2. Save PDF locally and Create unique paths for FAISS and BM25 models
        filename = secure_filename(pdf_file.filename)
        base_name = os.path.splitext(filename)[0]
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        pdf_file.save(pdf_path)

        faiss_path = os.path.join(MODEL_FOLDER, f"{base_name}_faiss_index")
        bm25_path = os.path.join(MODEL_FOLDER, f"{base_name}_bm25_index")

        # 3. Extract text from PDF
        raw_pages = extract_text_from_pdf(pdf_path) 
        print("Extraction completed") 

        # 4. Split text into chunks
        docs = split_text(raw_pages,config.CHUNK_SIZE,config.CHUNK_OVERLAP)  # Returns list of Document objects
        print("Chunking Completed")

        # 5. Build FAISS index
        vectorstore = build_faiss_index(docs)  
        print("Build Complete for FAISS")
        # 6. Build BM25 index
        bm25, corpus = build_bm25_index(docs) 
        print("Build Complete for BM25")

        # 7. Save both models
        save_faiss(vectorstore, faiss_path)  
        save_bm25(bm25, corpus, bm25_path) 
        print("Save both models")

        # 8. Save mapping
        save_model_mapping_to_excel(filename, faiss_path, bm25_path) 
        print("Mapping Completed")

        return jsonify({"status": "success", "message": "Models saved successfully"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8077)

"""
Hackathon Toolkit - Ready-to-use AI snippets for fast prototyping
Author: You :)
"""

# 🔹 Core Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hugging Face
from transformers import pipeline

# LangChain + OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Diffusers for Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

# OpenAI API (Speech)
import openai

# =============================
# 🔹 TEXT TASKS
# =============================

def sentiment_analysis(text, model="distilbert-base-uncased-finetuned-sst-2-english"):
    clf = pipeline("text-classification", model=model)
    return clf(text)

def named_entity_recognition(text, model="dslim/bert-base-NER"):
    ner = pipeline("ner", model=model, aggregation_strategy="simple")
    return ner(text)

def question_answering(question, context, model="distilbert-base-uncased-distilled-squad"):
    qa = pipeline("question-answering", model=model)
    return qa(question=question, context=context)

def summarization(text, model="facebook/bart-large-cnn", max_length=50, min_length=10):
    summarizer = pipeline("summarization", model=model)
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

# =============================
# 🔹 RAG (Retrieval-Augmented Generation)
# =============================
def rag_pipeline(docs, query, embedding_model="text-embedding-3-small", llm_model="gpt-4o-mini"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    db = FAISS.from_texts(docs, embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model=llm_model)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)

# =============================
# 🔹 VISION TASKS
# =============================

def image_classification(image_path_or_url, model="google/vit-base-patch16-224"):
    clf = pipeline("image-classification", model=model)
    return clf(image_path_or_url)

def text_to_image(prompt, model="runwayml/stable-diffusion-v1-5", out_file="generated.png"):
    pipe = StableDiffusionPipeline.from_pretrained(model)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt).images[0]
    image.save(out_file)
    return out_file

# =============================
# 🔹 SPEECH
# =============================

def speech_to_text(audio_file, model="gpt-4o-mini-transcribe", api_key=None):
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcriptions.create(model=model, file=f)
    return transcript.text

def text_to_speech(text, model="gpt-4o-mini-tts", voice="alloy", out_file="speech.mp3", api_key=None):
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    response = openai.audio.speech.create(model=model, voice=voice, input=text)
    with open(out_file, "wb") as f:
        f.write(response.read())
    return out_file

# =============================
# 🔹 ML (Tabular Data)
# =============================

def preprocess_data(df, target_col):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_col, axis=1), df[target_col], test_size=0.2
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf

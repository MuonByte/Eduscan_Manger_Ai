import os
import re
import glob
import shutil
import nltk
import torch
import numpy as np
from PIL import Image
from autocrop import Cropper
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# ====== إعدادات النماذج ======
GPT_MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_NAME)

MODEL_DIR = "models/all-MiniLM-L6-v2"
if not os.path.exists(MODEL_DIR):
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=MODEL_DIR)
EMBEDDING_MODEL = SentenceTransformer(MODEL_DIR)

nltk.download('punkt', quiet=True)

# ====== إعدادات RAG ======
INDEX_DIR = 'rag_index'
client = Client(Settings(persist_directory=INDEX_DIR, anonymized_telemetry=False))
collection = client.get_or_create_collection('edu_docs')

# ====== أدوات OCR وغيرها ======
def auto_crop_image(image_path):
    try:
        cropper = Cropper()
        cropped = cropper.crop(image_path)
        return Image.fromarray(cropped) if cropped is not None else Image.open(image_path)
    except:
        return Image.open(image_path)

def run_ocr(path):
    doc = DocumentFile.from_pdf(path) if path.lower().endswith('.pdf') else DocumentFile.from_images(path)
    predictor = ocr_predictor(pretrained=True)
    return predictor(doc)

def extract_text(result):
    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
    return "\n".join(lines)

def generate_gpt_neo_response(prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def correct_text(text):
    prompt = f"Correct the spelling and grammar of this educational content:\n{text}"
    return generate_gpt_neo_response(prompt)

def summarize_text(text):
    prompt = (
        "Please provide a clear and concise summary of the following educational content:\n\n"
        f"{text}\n\nSummary:"
    )
    return generate_gpt_neo_response(prompt)

def segment_text(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    headings = [s for s in sentences if s.isupper()]
    bullets = [s for s in sentences if re.match(r'^[-*•]', s.strip())]
    other = [s for s in sentences if s not in headings and s not in bullets]
    return {"headings": headings, "bullets": bullets, "other_text": other}

def build_or_update_index(folder='output'):
    docs = []
    for file in glob.glob(os.path.join(folder, '*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            docs.append({'id': file, 'text': f.read()})
    if docs:
        embeddings = [EMBEDDING_MODEL.encode(d['text']) for d in docs]
        collection.add(
            documents=[d['text'] for d in docs],
            metadatas=[{'source': d['id']} for d in docs],
            ids=[d['id'] for d in docs],
            embeddings=embeddings
        )

def retrieve_passages(query, top_k=5):
    try:
        q_emb = EMBEDDING_MODEL.encode(query)
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        return results['documents']
    except:
        return []

def ask_gpt_neo(query, context_passages):
    combined = "\n<<<CONTEXT>>>\n" + "\n---\n".join(p[0] if isinstance(p, list) else p for p in context_passages)
    prompt = (
        "You are an expert assistant. Based on the context, answer the question below.\n\n"
        f"Context:\n{combined}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return generate_gpt_neo_response(prompt)

# ====== API: رفع ملف PDF أو صورة متعددة الصفحات ======
@app.post("/api/ai/upload-file")
async def process_file(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # OCR شامل كل الصفحات
    result = run_ocr(file_path)
    raw_text = extract_text(result)

    corrected = correct_text(raw_text)
    summary = summarize_text(corrected)
    segments = segment_text(corrected)

    # حفظ في ملف
    out_path = os.path.join("output", file.filename + ".txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(corrected)

    build_or_update_index()

    return JSONResponse({
        "raw_text": raw_text,
        "corrected_text": corrected,
        "summary": summary,
        "segments": segments,
        "original":corrected,
        "augmented":segments
    })

# ====== API: سؤال باستخدام RAG ======
@app.post("/api/ai")
async def ask_question(prompt:str):
    prompt = prompt
    passages = retrieve_passages(prompt)
    answer = ask_gpt_neo(prompt, passages)
    return {"question": prompt, "answer": answer}

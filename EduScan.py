import os
import re
import glob
import nltk
import cv2
import tkinter as tk
import inspect
from tkinter import filedialog
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import numpy as np
from autocrop import Cropper
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
GPT_MODEL_NAME = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_NAME)


MODEL_DIR = r"D:\Ai\Eduscan\models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_DIR):
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=MODEL_DIR
    )

EMBEDDING_MODEL = SentenceTransformer(MODEL_DIR)

nltk.download('punkt', quiet=True)

INDEX_DIR = 'rag_index'

def build_or_load_index(folder='output', persist_dir=INDEX_DIR):
    print("Starting to build or load RAG index...")
    docs = []
    for file in glob.glob(os.path.join(folder, '*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            docs.append({'id': file, 'text': f.read()})

    client = Client(Settings(
        persist_directory=persist_dir,
        anonymized_telemetry=False
    ))
    collection = client.get_or_create_collection('edu_docs')

    if collection.count() == 0 and docs:
        embeddings = [EMBEDDING_MODEL.encode(d['text']) for d in docs]
        collection.add(
            documents=[d['text'] for d in docs],
            metadatas=[{'source': d['id']} for d in docs],
            ids=[d['id'] for d in docs],
            embeddings=embeddings
        )
        print(f"Indexed {len(docs)} documents to '{persist_dir}'")
    else:
        print(f"Loaded existing index with {collection.count()} documents.")
    print("RAG index operation completed.")
    return collection

def select_files():
    print("Opening file selection dialog for multiple files...")
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Image or PDF files",
        filetypes=[("Image and PDF files", "*.jpg *.jpeg *.png *.pdf"), ("All files", "*.*")]
    )
    if file_paths:
        print(f"Selected files: {file_paths}")
    else:
        print("No files selected.")
    return list(file_paths)

def auto_crop_image(image_path):
    print(f"Starting auto-cropping for {image_path}...")
    try:
        cropper = Cropper()
        cropped_image = cropper.crop(image_path)
        if cropped_image is not None:
            if isinstance(cropped_image, np.ndarray):
                cropped_image = Image.fromarray(cropped_image)
            print("Image auto-cropped successfully.")
            return cropped_image
        else:
            print("Auto-cropping did not yield a result. Proceeding with original image.")
            return Image.open(image_path)
    except Exception as e:
        print(f"Error in auto-cropping: {e}. Proceeding with original image.")
        try:
            return Image.open(image_path)
        except Exception as e2:
            print(f"Error opening original image: {e2}")
            return None

def run_ocr(path):
    print(f"Running OCR on {path}...")
    try:
        if not os.path.exists(path):
            print(f"Error: File {path} does not exist.")
            return None
        if path.lower().endswith(('jpg', 'jpeg', 'png')):
            doc = DocumentFile.from_images(path)
        else:
            doc = DocumentFile.from_pdf(path)
        ocr_result = ocr_predictor(pretrained=True)(doc)
        print("OCR completed.")
        return ocr_result
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

def extract_text(result):
    print("Extracting text from OCR result...")
    try:
        if result is None:
            print("No OCR result to extract text from.")
            return ""
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))
        extracted_text = "\n".join(lines)
        if not extracted_text:
            print("Warning: No text extracted from OCR result.")
        else:
            print(f"Extracted text: {extracted_text[:100]}...")  
        return extracted_text
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return ""

def generate_gpt_neo_response(prompt, max_tokens=150):
    print("Generating response using GPT-Neo...")
    try:
        inputs = tokenizer(prompt, return_tensors="pt")  # تعريف inputs مفقود
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response[:100]}...")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}"

def correct_text(text):
    print("Correcting text using GPT-Neo...")
    if not text:
        print("Warning: No text provided for correction.")
        return ""
    prompt = f"Correct the spelling and grammar of this educational content:\n{text}"
    return generate_gpt_neo_response(prompt)

def summarize_text(text):
    print("Summarizing content using GPT-Neo...")
    if not text:
        print("Warning: No text provided for summarization.")
        return ""
    prompt = (
        "Please provide a clear and concise summary of the following educational content, "
        "highlighting the main points and key information:\n\n"
        f"{text}\n\nSummary:"
    )
    
    return generate_gpt_neo_response(prompt, max_tokens=150)

def ask_gpt_neo(query, context_passages):
    print("Asking GPT-Neo with RAG context...")

    # تأكد أن كل عنصر في context_passages هو نص (str)
    # إذا كان فيه قوائم، ناخذ أول عنصر منها (كما في كودك الأصلي)
    clean_passages = []
    for p in context_passages:
        if isinstance(p, list) and len(p) > 0:
            clean_passages.append(p[0])
        elif isinstance(p, str):
            clean_passages.append(p)
        else:
            # تجاهل أي عناصر غير نصية أو فارغة
            continue

    if clean_passages:
        combined = "\n<<<CONTEXT>>>\n" + "\n---\n".join(clean_passages)
    else:
        combined = "No context available."

    print(f"Combined context (preview):\n{combined[:500]}")  # لعرض جزء من المحتوى للتأكد

    prompt = (
        "You are an expert educational assistant. You have the following context passages.\n"
        "First, review each passage and identify which ones are most relevant to the user's question.\n"
        "Then, provide a detailed and accurate answer using only the most relevant information.\n"
        "If the answer is not contained within the context, say 'The information is not in the context, but I will try to answer based on my knowledge.' and then provide a helpful answer.\n\n"
        f"Context passages:\n{combined}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )

    return generate_gpt_neo_response(prompt)


def tag_by_subject(text, subject_name):
    print(f"Tagging text with subject: {subject_name}...")
    tagged_text = f"\n{text}"
    print("Text tagging completed.")
    return tagged_text

def segment_text(text):
    print("Segmenting text into sections...")
    if not text:
        print("Warning: No text provided for segmentation.")
        return {"headings": [], "bullets": [], "other_text": []}
    sentences = nltk.tokenize.sent_tokenize(text)
    headings = [s for s in sentences if s.isupper()]
    bullets = [s for s in sentences if re.match(r'^[-*•]', s.strip())]
    other = [s for s in sentences if s not in headings and s not in bullets]
    print("Text segmentation completed.")
    return {"headings": headings, "bullets": bullets, "other_text": other}


def save_processed_outputs(summary):
    print("Saving processed outputs...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    if not summary:
        print("Warning: Summary is empty, writing empty summary.md")
        summary = "No summary generated."
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as sum_file:
        sum_file.write(f"# Summary of \n\n{summary}\n")
    print("[✅] Outputs saved successfully in folders.")

def retrieve_passages(query, collection, top_k=7):
    print(f"Retrieving top {top_k} passages for query: '{query}'...")
    try:
        q_emb = EMBEDDING_MODEL.encode(query)
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        print("Passage retrieval completed.")
        return results['documents']
    except Exception as e:
        print(f"Error in passage retrieval: {e}")
        return []


def save_chat_log(query, answer, log_directory="chat_logs", log_filename="chat_history.txt"):
    print(f"Saving chat log to {log_directory}/{log_filename}...")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_directory)
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_filename)
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"User: {query}\n")
            f.write(f"Assistant: {answer}\n\n")
        print("[✅] Chat log saved successfully.")
    except PermissionError:
        print(f"Error: Insufficient file permissions to write to {file_path}. Please check directory permissions.")
    except IOError as e:
        print(f"An I/O error occurred while saving chat log: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving chat log: {e}")

def main():
    print("📌 Starting main application flow...")
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    collection = build_or_load_index()
    paths = select_files()

    if not paths:
        print("No file(s) selected or process was canceled. Exiting.")
        return

    for path in paths:
        print(f"Processing file: {path}")
        if path.lower().endswith(('jpg', 'png', 'jpeg')):
            cr = auto_crop_image(path)
            if cr:
            
                if cr.mode == 'RGBA':
                    cr = cr.convert('RGB')
                cr.save('cropped_temp.jpg')
                path = 'cropped_temp.jpg'
                print("Image was auto-cropped and saved as cropped_temp.jpg.")
            else:
                print(f"Warning: Failed to process image {path}. Skipping.")
                continue

        ocr_res = run_ocr(path)
        raw_text = extract_text(ocr_res)
        if not raw_text:
            print(f"Warning: No text extracted from {path}. Skipping further processing.")
            continue

        corrected = correct_text(raw_text)
        if corrected.startswith("Error:"):
            print(f"Warning: Text correction failed: {corrected}")
            corrected = raw_text  

        os.makedirs('output', exist_ok=True)
        fname = os.path.basename(path) + '.txt'
        with open(os.path.join('output', fname), 'w', encoding='utf-8') as f:
            f.write(corrected if corrected else "No corrected text.")
        print(f"Saved corrected text to {fname}")

        collection = build_or_load_index()
        print("[✅] Content ingested.")

        summary = summarize_text(corrected)
        if summary.startswith("Error:"):
            print(f"Warning: Summarization failed: {summary}")
            summary = "Summary could not be generated."

        save_processed_outputs(summary)

        with open(os.path.join(output_dir, 'raw_text.txt'), 'w', encoding='utf-8') as f:
            f.write(raw_text if raw_text else "No text extracted.")
        with open(os.path.join(output_dir, 'corrected_text.txt'), 'w', encoding='utf-8') as f:
            f.write(corrected if corrected else "No corrected text.")
        with open(os.path.join(output_dir, 'summarize_text.txt'), 'w', encoding='utf-8') as f:
            f.write(summary if summary else "No summary generated.")
        with open(os.path.join(output_dir, 'ask_gpt_neo.txt'), 'w', encoding='utf-8') as f:
            f.write(inspect.getsource(ask_gpt_neo))

    chat_history = []
    while True:
        q = input("Ask Any question (or 'exit'): ")
        if q.lower() == 'exit':
            print("Exiting interactive chat.")
            break
        passages = retrieve_passages(q, collection)
        answer = ask_gpt_neo(q, passages)
        print("Assistant:", answer)
        chat_history.append(f"### User:\n{q}\n\n### Assistant:\n{answer}\n")
        save_chat_log(q, answer)

    with open(os.path.join(output_dir, 'chat_history.md'), 'w', encoding='utf-8') as f:
        for entry in chat_history:
            f.write(entry + "\n---\n")

    print("📌 Main application flow finished.")

if __name__ == '__main__':
    main()
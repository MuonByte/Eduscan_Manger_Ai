# EduScan – Intelligent Study Scanner & Tutor

EduScan is an AI-powered educational assistant that enables students to scan handwritten or printed materials and instantly convert them into structured, interactive learning content. It integrates OCR (Optical Character Recognition), text correction, summarization, and Retrieval-Augmented Generation (RAG) to create a seamless, AI-driven study experience.

## 🔧 Features

- 📸 **Image and Document Upload**: Upload handwritten or printed study materials as images (JPG/PNG) or PDFs.
- 🧠 **OCR & Text Extraction**: Automatically detect and extract text using Tesseract OCR.
- ✍️ **Text Correction**: Clean and correct grammar or spelling errors using AI-based tools.
- 📄 **Summarization**: Generate concise summaries of long documents or notes.
- 🗂️ **RAG (Retrieval-Augmented Generation)**: Index the extracted knowledge and allow users to query it using natural language.
- 🤖 **Local Mistral Integration**: Uses [https://lmstudio.ai/models/mistralai/mistral-7b-instruct-v0.3] via LM Studio for local inference and private querying.
- 🌐 **FastAPI Backend**: REST API for document upload, summarization, and querying.

## 📁 Project Structure

```
project_root/
├── main.py                  # Main script that controls workflow
├── image_utils.py           # File/image selection, autocrop, OCR
├── text_utils.py            # Text correction, summarization, segmentation
├── index_utils.py           # RAG index build/load and retrieval
├── chat_utils.py            # Chat interaction and logging
├── mistral_api.py           # Calls to local Mistral model via HTTP
├── api.py                   # FastAPI app for OCR + RAG + Q&A service
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```  

## 🚀 How to Run Locally

### ✅ `main.py`
Entry point of the application. Calls all components step-by-step. Manages file processing and chat loop.

---

### 🖼️ `image_utils.py`
- `select_files()`: Open dialog to select PDF/images
- `auto_crop_image(path)`: Crop image automatically
- `run_ocr(path)`: OCR via doctr
- `extract_text(result)`: Get plain text from OCR result

---

### 📝 `text_utils.py`
- `correct_text(text)`: Enhances and fixes raw OCR
- `summarize_text(text)`: Produces detailed summary
- `tag_by_subject(text, subject)`: Adds subject info
- `segment_text(text)`: Splits content into headings/bullets

---

### 📚 `index_utils.py`
- `build_or_load_index(folder, persist_dir)`: Build or load document embedding index
- `retrieve_passages(query, collection, top_k)`: Retrieve related passages from vector store

---

### 💬 `chat_utils.py`
- `save_chat_log(query, answer, log_directory, log_filename)`: Save Q&A into markdown chat log

---

### 🤖 `mistral_api.py`
1. **Download [LM Studio](https://lmstudio.ai/)**
  2. Open LM Studio, then **download the model**: `mistralai/mistral-7b-instruct-v0.3` from the "Models" section.
  3. Go to the **"Server" tab** and click **“Start Server”**. This launches a local API endpoint.
  4. ![LM](https://github.com/user-attachments/assets/8f097692-0fc3-4833-ba31-9ace495ccf91)

  5. Inside your code ( `mistral_api.py`), make sure the `url` is set to your local endpoint, for example:
     ```python
     url = "http://Your Local server/v1/chat/completions"
     ```
  6. Your system is now ready to make local Mistral API requests, ensuring **full privacy and low latency** without cloud-based models.
- `generate_mistral_response(prompt, max_tokens)`: Send user prompt to Mistral API (local LM Studio)
- `ask_mistral(query, context)`: Combines context with query and gets answer from Mistral

---

### 🌐 `api.py`
FastAPI-based web API to:
- Upload and process educational documents via `/upload`
- Run RAG-based Q&A via `/query`
- Health check endpoint `/health`

---





## 👩‍💻 Author

Made with ❤️ by [Zeyad Alaa & Mahmoud Gomaa] – Cairo, Egypt  
For feedback or collaboration, feel free to reach out!

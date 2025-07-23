# PEFT_RAG

This repository contains a collection of tools and notebooks for Retrieval-Augmented Generation (RAG) with Parameter-Efficient Fine-Tuning (PEFT) techniques. It includes:

- **A Streamlit app for chatting with PDF documents using RAG and Google Gemini**
- **Notebooks for LoRA/QLoRA fine-tuning and inference on large language models**
- **A notebook for fine-tuning T5 for conversational AI**

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Notebooks Overview](#notebooks-overview)
- [License](#license)

---

## Features

- **Chat with PDFs**: Upload PDF files and interact with them using a conversational AI powered by Google Gemini and FAISS vector search.
- **RAG Pipeline**: Extracts, chunks, embeds, and indexes PDF content for context-aware Q&A.
- **LoRA/QLoRA Fine-Tuning**: Parameter-efficient fine-tuning of large language models using Hugging Face Transformers and PEFT.
- **Conversational T5 Fine-Tuning**: End-to-end notebook for fine-tuning T5 on conversational datasets.

---

## Project Structure

```
RAG_PEFT/
│
├── RAG.py                                 # Streamlit app for PDF chat with RAG
├── final-inference-gpu-test-qlora.ipynb   # QLoRA inference and fine-tuning notebook
├── final-lora.ipynb                       # LoRA fine-tuning notebook
├── fine-tune-t5-for-conversational-model.ipynb # T5 conversational fine-tuning
├── README.md                              # Project documentation
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PujanMevawala/PEFT_RAG.git
cd PEFT_RAG
```

### 2. Install Python Dependencies

Install required packages for the Streamlit app and notebooks:

```bash
pip install -r requirements.txt
# Or install manually:
pip install streamlit langchain PyPDF2 faiss-cpu google-generativeai langchain-google-genai python-dotenv
```

For notebooks, install additional packages as needed (see notebook cells):

```bash
pip install transformers accelerate peft datasets bitsandbytes pytorch-lightning wandb
```

### 3. Set Up Environment Variables

Create a `.env` file and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Usage

### Run the Streamlit PDF Chat App

```bash
streamlit run RAG.py
```

1. Upload one or more PDF files in the sidebar.
2. Click "Submit & Process" to index the documents.
3. Ask questions about the PDFs in the chat interface.

### Run the Notebooks

Open any notebook (`.ipynb`) in Jupyter or VS Code and follow the instructions in the cells to perform fine-tuning or inference.

---

## Notebooks Overview

- **final-inference-gpu-test-qlora.ipynb**: QLoRA-based inference and fine-tuning for large models using Hugging Face and PEFT.
- **final-lora.ipynb**: LoRA fine-tuning workflow for transformer models.
- **fine-tune-t5-for-conversational-model.ipynb**: Fine-tune T5 for conversational tasks using PyTorch Lightning.

---

## License

This project is licensed under the MIT License.

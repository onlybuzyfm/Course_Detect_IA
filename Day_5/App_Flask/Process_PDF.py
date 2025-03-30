import os
import zipfile
import fitz  # PyMuPDF
import re
import math
import numpy as np
from collections import Counter
import pandas as pd
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ----------------------------
# Cargar modelo GPT-2 una vez
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        return math.exp(loss.item())

# ----------------------------
# Procesamiento de PDFs
# ----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def entropy(text):
    probs = [freq / len(text) for freq in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def burstiness(text):
    words = text.split()
    word_counts = Counter(words)
    freqs = list(word_counts.values())
    if len(freqs) < 2:
        return 0.0
    return np.std(freqs) / np.mean(freqs)

def extract_zip(zip_path, extract_to="pdfs_temp"):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    pdf_files = []
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    return pdf_files

def process_pdfs_in_zip(zip_path, extract_to="pdfs_temp"):
    pdf_paths = extract_zip(zip_path, extract_to=extract_to)
    results = []

    for path in pdf_paths:
        print(f"Procesando: {os.path.basename(path)}")
        raw = extract_text_from_pdf(path)
        clean = preprocess_text(raw)
        ent = entropy(clean)
        bur = burstiness(clean)
        ppl = calculate_perplexity(clean)

        results.append({
            "archivo": os.path.basename(path),
            "entropia": round(ent, 4),
            "burstiness": round(bur, 4),
            "perplejidad": round(ppl, 4)
        })

    return pd.DataFrame(results)

# ----------------------------
# Detección de sospechosos
# ----------------------------
def calcular_umbral_penalizacion(df, sensibilidad=1.0):
    umbrales = {}
    for columna in ["entropia", "burstiness", "perplejidad"]:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - sensibilidad * iqr
        lim_sup = q3 + sensibilidad * iqr

        umbrales[columna] = {
            "q1": q1,
            "q3": q3,
            "inferior": lim_inf,
            "superior": lim_sup
        }

    return umbrales

def marcar_sospechosos_compuesto(df, ppl_ratio=0.6, entropia_min=3.5):
    max_ppl = df["perplejidad"].max()
    df["perplejidad_relativa"] = df["perplejidad"] / max_ppl

    def sospechoso(row):
        condiciones = [
            row["perplejidad_relativa"] < ppl_ratio,
            row["entropia"] < entropia_min
        ]
        return "Si" if sum(condiciones) >= 1 else "No"

    df["sospechoso"] = df.apply(sospechoso, axis=1)
    return df


# A Comprehensive Evaluation of Large Language Models for Information Extraction from Unstructured Electronic Health Records in Residential Aged Care
Dinithi Vithanage1, Ping Yu1, *, Qianqian Xie2, Hua Xu2 Lei Wang1, Chao Deng3 

1	School of Computing and Information Technology, University of Wollongong, Wollongong, Australia
2	Department of Biomedical Informatics and Data Science, Yale University, USA.
3	School of Medical, Indigenous and Health Sciences, University of Wollongong, Wollongong, Australia


This repository contains all necessary components to implement and evaluate a Retrieval-Augmented Generation (RAG) pipeline using large language models (LLMs) for information extraction from unstructured Electronic Health Records (EHRs) in Residential Aged Care.
##Citation: Vithanage, D., Yu, P., Xie, Q., Xu, H., Wang, L., Deng, C. (2025) A comprehensive evaluation of large language models for information extraction from unstructured electronic health records in residential aged care. Computers in Biology and Medicine 197(Pt A): p. 111013.



## 🔍 Overview

This project explores and compares multiple configurations of RAG pipelines using general and domain-specific LLMs. It supports modular execution via:

- `LangChain`  
- `LlamaIndex`  
- `LangChain + LlamaIndex`

The framework enables experimentation with different retrieval backends, prompting strategies, and model types to assess their effectiveness in extracting relevant clinical information.

---

## 📁 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `RAGKnowledgeFiles/` | Domain-specific clinical documents used for retrieval. |
| `PromptTemplates/`   | Prompt templates to instruct the language model. |
| `LangChain.py`       | Implements the RAG pipeline using LangChain. |
| `Llama_index.py`     | Implements the RAG pipeline using LlamaIndex. |
| `Llama_index+Lang_Chain.py` | Combines LangChain and LlamaIndex in the RAG pipeline. |
| `Evaluation.py`      | Script to compute evaluation metrics (F1, Fairness, Bias, Robustness, ROUGE, BERTScore, BLEU, BLEURT, METEOR). |

---

## ⚙️ How to Use

### 1. Setup Environment

Install the required packages:

```bash
pip install llama-index transformers langchain torch bert-score rouge-score nltk bleurt pandas
```

Then run once to download NLTK resources:

```python
import nltk
nltk.download('punkt')
```

### 2. Prepare Files

Upload the following into your working directory:

- `RAGKnowledgeFiles/`
- `PromptTemplates/`
- `test_data.csv`
- `generated_answers.csv` – model outputs
- `reference_answers.csv` – annotated ground truth for evaluation

> ⚠️ **Note**: The actual test data (`test_data.csv`, `reference_answers.csv`, and `generated_answers.csv`) is **not included** due to regulatory and ethical restrictions. Please use your own dataset.

---

### 3. Run RAG Pipeline

Choose and run **one** of the RAG pipeline implementations according to your requirements:

```bash
python LangChain.py
python Llama_index.py
python Llama_index+Lang_Chain.py
```

All scripts read from the knowledge and prompt directories, and generate model outputs to `generated_answers.csv`.

---

### 4. Run Evaluation

Evaluate model outputs using the following command:

```bash
python Evaluation.py
```

---

## 🧠 Model Flexibility

You are free to use any HuggingFace-compatible model. Simply modify the model name in the RAG pipeline scripts to load:
- General-purpose LLMs (e.g., `gpt2`, `llama-2`, `mistral`)
- Biomedical LLMs (e.g., `BioGPT`, `ClinicalBERT`, `PubMedBERT`)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## 📞 Contact

For questions or reproduction guidance, please refer to the corresponding research article or contact the authors.

---

## 📜 Citation

If you use this work, please cite the accompanying research paper (details in publication).

---


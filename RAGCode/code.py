# -*- coding: utf-8 -*-
"""Code.ipynb

"""

# Import necessary libraries and modules for building and querying the LlamaIndex pipeline
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

# Load documents from the local directory "RAGKnowledgeFiles"
reader = SimpleDirectoryReader(input_dir="RAGKnowledgeFiles")
documents = reader.load_data()

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Import necessary libraries
import torch
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
# Initialize the HuggingFaceLLM class with model and generation configurations
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="",
    model_name="",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding

# Load the prompt from a separate text file
with open("your_prompt_file.txt", "r") as f:
    prompt = f.read()

# Define the embedding model
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="your-model-name")  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
)

# Create the service context
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Build the index from documents
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Create the query engine and run the query
query_engine = index.as_query_engine()
response = query_engine.query(prompt)
print(response)

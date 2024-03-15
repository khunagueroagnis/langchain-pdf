import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = ""

import textract
doc = textract.process("./The last wish.pdf")

with open('The last wish.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('The last wish', 'r') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))    

#splitting text into chunks

text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

type(chunks[0]) 


# embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

query = "What did gerlat wish for?"
docs = db.similarity_search(query)
docs[0]



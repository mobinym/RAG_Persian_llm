import os
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# مسیر فایل ورد شما
DOCUMENT_PATH = r"C:\Users\m.yaghoubi\Desktop\rag\adamiyan_[www.ketabesabz.com].docx" 
LOCAL_LLM_NAME = "gemma3" 

if not os.path.exists(DOCUMENT_PATH):
    print(f"Error: File not found at path '{DOCUMENT_PATH}'.")
    exit()

print("1. Starting to process the Word document...")

loader = Docx2txtLoader(DOCUMENT_PATH)
docs = loader.load()

print("... Cleaning the text from extra information ...")
original_text = docs[0].page_content
cleaned_text = re.sub(r'www.ketabesabz.com', '', original_text)
cleaned_text = re.sub(r'صفحه: \d+', '', cleaned_text)
cleaned_text = re.sub(r'آدمیان \d+ ‎\d+‏', '', cleaned_text)
cleaned_text = re.sub(r'\d+ #? ?زویا قلی‌پور', '', cleaned_text)
cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
docs[0].page_content = cleaned_text 

print(f"Document loaded and cleaned successfully. Character count: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"Text has been split into {len(chunks)} chunks.")

print("... Creating embeddings with a powerful multilingual model ...")

# --- تغییر اصلی و نهایی اینجاست ---
# استفاده از یک مدل استاندارد و بسیار قدرتمند چندزبانه
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# نکته: این مدل کمی بزرگتر است و دانلود اولیه آن ممکن است چند دقیقه طول بکشد، اما ارزشش را دارد.

vectorstore = FAISS.from_documents(chunks, embedding_model)
print("Vector store created successfully.")

print("... Creating an advanced retriever with a Re-ranker ...")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
compressor = FlashrankRerank(top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

print("... Setting up the Language Model (LLM) and prompt ...")
llm = ChatOllama(model=LOCAL_LLM_NAME, temperature=0.1) 

PROMPT_TEMPLATE = """
شما یک دستیار هوش مصنوعی هستید که وظیفه‌تان پاسخ به سوالات بر اساس یک متن مشخص است.
فقط و فقط بر اساس متن ارائه شده در بخش «زمینه» به سوال کاربر در بخش «سوال» پاسخ بده.
پاسخ‌هایت باید کامل، دقیق و تمیز باشند.
از دانش قبلی خود به هیچ وجه استفاده نکن.
اگر پاسخ سوال در متن «زمینه» وجود نداشت، به صورت واضح و مودبانه بگو: "پاسخ این سوال در سند موجود نیست."

زمینه:
{context}

سوال:
{question}

پاسخ دقیق و کامل:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ System is ready. Please ask your question.")
print("Type 'exit' or 'خروج' to quit.")
print("-" * 50)

while True:
    user_question = input("You: ")
    if user_question.lower() in ["خروج", "exit"]:
        break
    
    print("Searching, re-ranking, and generating a response...")
    response = rag_chain.invoke(user_question)
    
    print("\nSystem's Answer:")
    print(response)
    print("-" * 50)
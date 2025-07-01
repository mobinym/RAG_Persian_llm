import os
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- بخش تنظیمات ---
# مسیر فایل ورد خود را اینجا قرار دهید
DOCUMENT_PATH = r"C:\Users\m.yaghoubi\Desktop\rag\adamiyan_[www.ketabesabz.com].docx" 
# نام بهترین مدل Embedding
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# نام مدل LLM شما در Ollama
LOCAL_LLM_NAME = "gemma3"

# --- ۱. بارگذاری و پاک‌سازی سند ---
if not os.path.exists(DOCUMENT_PATH):
    print(f"Error: The file '{DOCUMENT_PATH}' does not exist.")
    exit()

print("1. Loading and cleaning the document...")
loader = Docx2txtLoader(DOCUMENT_PATH)
documents = loader.load()

# اعمال پاک‌سازی برای تضمین کیفیت ورودی
original_text = documents[0].page_content
cleaned_text = re.sub(r'www\.ketabesabz\.com', '', original_text, flags=re.IGNORECASE)
cleaned_text = re.sub(r'صفحه: \d+', '', cleaned_text)
cleaned_text = re.sub(r'آدمیان \d+ ‎\d+‏', '', cleaned_text)
cleaned_text = re.sub(r'\d+ #? ?زویا قلی‌پور', '', cleaned_text)
cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
documents[0].page_content = cleaned_text
print("Document loaded and cleaned successfully.")

# --- ۲. چانکینگ ترکیبی (مبتنی بر توکن و بازگشتی) ---
print("2. Chunking the document using token-based recursive splitting...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",  # استفاده از توکنایزر استاندارد به عنوان معیار دقیق
    chunk_size=300,      # اندازه هر قطعه بر اساس توکن
    chunk_overlap=60     # همپوشانی بین قطعات بر اساس توکن
)
chunks = text_splitter.split_documents(documents)
print(f"Document split into {len(chunks)} chunks.")

# --- ۳. ساخت Embedding و Vector Store ---
print(f"3. Creating embeddings using the powerful '{EMBEDDING_MODEL_NAME}' model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = FAISS.from_documents(chunks, embedding_model)
print("Vector store created successfully.")

# --- ۴. ساخت Retriever پیشرفته با Re-ranker ---
print("4. Creating an advanced retriever with a Re-ranker...")
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
compressor = FlashrankRerank(top_n=3)
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# --- ۵. تعریف LLM و پرامپت ---
llm = OllamaLLM(model=LOCAL_LLM_NAME, temperature=0.1)

prompt_template = """شما یک دستیار هوش مصنوعی دقیق و وظیفه‌شناس هستید. وظیفه شما پاسخ به سوال کاربر فقط و فقط بر اساس اطلاعاتی است که در بخش "زمینه" به شما داده می‌شود.
پاسخ شما باید کامل، دقیق و مستقیماً از متن استخراج شده باشد. از دانش قبلی خود به هیچ عنوان استفاده نکنید.
اگر پاسخ سوال در متن "زمینه" وجود نداشت، به صورت واضح و مودبانه بگویید: "پاسخ این سوال در سند موجود نیست."

زمینه:
{context}

سوال:
{question}

پاسخ دقیق:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- ۶. ساخت زنجیره مدرن با LCEL ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ System is fully initialized and ready. You can ask your questions.")
print("Type 'exit' or 'خروج' to quit.")
print("-" * 50)

# --- ۷. حلقه تعاملی پرسش و پاسخ ---
while True:
    user_question = input("You: ")
    if user_question.lower() in ["خروج", "exit"]:
        break
    
    print("Searching, re-ranking, and generating a response...")
    result = rag_chain.invoke(user_question)
    
    print("\nSystem's Answer:")
    print(result)
    print("-" * 50)
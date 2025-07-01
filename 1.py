import re
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from datetime import datetime

# --- استخراج متن از فایل ورد ---
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
    return text

# --- پاکسازی متن فارسی ---
def clean_persian_text(text):
    text = re.sub(r'[\u200c\u200d\u200e\u200f]', '', text)  # حذف کاراکترهای نامرئی
    text = re.sub(r'\s+', ' ', text)  # حذف فاصله‌های اضافی
    return text.strip()

# --- تقسیم متن به قطعات کوچک ---
def split_text(text, max_length=500):
    sentences = re.split(r'(?<=[.!؟])\s+', text)  # جداکردن بر اساس نقطه یا علامت سوال
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_length:
            current += s + " "
        else:
            chunks.append(current.strip())
            current = s + " "
    if current:
        chunks.append(current.strip())
    return chunks

# --- ساخت ایندکس Faiss ---
def build_faiss_index(chunks, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index, chunks

# --- بازیابی بخش‌های مرتبط ---
def ask_question(question, model, index, chunks, top_k=5):
    q_emb = model.encode([question])
    distances, indices = index.search(np.array(q_emb), top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return "\n---\n".join(retrieved)

# --- ارسال پرامپت به مدل LLM با محدودیت سختگیرانه ---
def generate_answer(query, context, model_name="gemma3"):
    prompt = f"""
فقط و فقط با استفاده از متن زیر به سؤال پاسخ بده. اگر جواب در متن نیست، بنویس: «در متن اشاره‌ای نشده است.»
هیچ اطلاعات اضافه و خارج از متن نگو.

--- متن ---
{context}
-------------

سؤال: {query}

پاسخ (فقط از متن بالا):
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# --- ثبت لاگ کامل ---
def log_interaction(question, context, answer, log_file="rag_log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n====== {datetime.now()} ======\n")
        f.write(f"❓ سؤال:\n{question}\n")
        f.write(f"📚 متن بازیابی‌شده:\n{context}\n")
        f.write(f"✅ پاسخ مدل:\n{answer}\n")
        f.write("====================================\n")

# --- بررسی اعتبار پاسخ بر اساس کلمات غیرمجاز ---
def check_answer_validity(answer, context, blacklist_words=None):
    if blacklist_words is None:
        blacklist_words = ["پلیس", "خانه", "مبارزه", "رقیب", "ماشین", "آماده", "خونه", "بیرون", "خارج", "ماشین پلیس"]
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    for word in blacklist_words:
        if word in answer_lower and word not in context_lower:
            return False, word
    return True, None

# --- حلقه اصلی پرسش و پاسخ ---
if __name__ == "__main__":
    file_path = r"C:\Users\m.yaghoubi\Desktop\rag\adamiyan_[www.ketabesabz.com].docx"  # مسیر فایل docx خودت

    raw_text = extract_text_from_docx(file_path)
    clean_text = clean_persian_text(raw_text)
    chunks = split_text(clean_text)
    model, index, chunks = build_faiss_index(chunks)

    while True:
        user_question = input("\n🔹 سؤال خود را وارد کن (برای خروج بنویس exit):\n")
        if user_question.strip().lower() == "exit":
            print("خداحافظ!")
            break

        context = ask_question(user_question, model, index, chunks)
        answer = generate_answer(user_question, context)

        valid, bad_word = check_answer_validity(answer, context)
        retry_count = 0
        max_retries = 3

        while not valid and retry_count < max_retries:
            print(f"\n⚠️ پاسخ مدل شامل کلمه غیرمجاز '{bad_word}' است. دوباره تلاش می‌کنیم...")
            prompt = f"""
فقط و فقط با استفاده از متن زیر به سؤال پاسخ بده. اگر جواب در متن نیست، بنویس: «در متن اشاره‌ای نشده است.»
هیچ اطلاعات اضافه و خارج از متن نگو. کلمه '{bad_word}' را در جواب نیاور.

--- متن ---
{context}
-------------

سؤال: {user_question}

پاسخ (فقط از متن بالا):
"""
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3", "prompt": prompt, "stream": False}
            )
            answer = response.json()["response"]
            valid, bad_word = check_answer_validity(answer, context)
            retry_count += 1

        if not valid:
            answer = "⚠️ متأسفانه پاسخ دقیقی در متن پیدا نشد یا مدل قادر به تولید پاسخ دقیق نیست."

        print("\n✅ پاسخ نهایی:\n", answer)
        log_interaction(user_question, context, answer)

import os
import re
from langchain_community.document_loaders import Docx2txtLoader

# مسیر فایل ورد شما
DOCUMENT_PATH = r"C:\Users\m.yaghoubi\Desktop\rag\adamiyan_[www.ketabesabz.com].docx"
# نام فایل خروجی برای بررسی
OUTPUT_DEBUG_FILE = "debug_output.txt"

print(f"Reading file from: {DOCUMENT_PATH}")

if not os.path.exists(DOCUMENT_PATH):
    print(f"Error: File not found at path '{DOCUMENT_PATH}'.")
    exit()

# فقط بارگذاری و پاک‌سازی
try:
    loader = Docx2txtLoader(DOCUMENT_PATH)
    docs = loader.load()

    original_text = docs[0].page_content
    cleaned_text = re.sub(r'www.ketabesabz.com', '', original_text)
    cleaned_text = re.sub(r'صفحه: \d+', '', cleaned_text)
    cleaned_text = re.sub(r'آدمیان \d+ ‎\d+‏', '', cleaned_text)
    cleaned_text = re.sub(r'\d+ #? ?زویا قلی‌پور', '', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

    # ذخیره کل متن پاک‌شده در یک فایل txt
    with open(OUTPUT_DEBUG_FILE, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
        
    print(f"\nSUCCESS: The entire cleaned text has been saved to the file: '{OUTPUT_DEBUG_FILE}'")
    print("Please open this file and check its content.")

except Exception as e:
    print(f"\nAn error occurred during file loading or processing: {e}")
import pytesseract
import os
import io
import random
import pdfplumber
import docx
import streamlit as st
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS
from transformers import pipeline
import re
import pandas as pd
import camelot
import tempfile

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
st.set_page_config(page_title="Document Analyzer", layout="centered")
load_dotenv()
spinner_placeholder = st.empty()

# --- Session State ---
for key, val in {
    'summary': '', 'language': 'en', 'last_uploaded_file': None,
    'editable_summary': '', 'summary_en': '', 'summary_mr': ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

LABELS = {
    "en": {
        "title": "📄 Document Analyzer",
        "subheading": "Summarize your documents and listen to them in voice",
        "upload": "Upload a document (PDF or DOCX)",
        "generate": "⚙️ Generate Summary",
        "read": "🔊 Read Summary",
        "switch_lang": "🔁 Switch Language (English/Marathi)",
        "summary_header": "📝 Summary"
    },
    "mr": {
        "title": "📄 दस्तऐवज विश्लेषक",
        "subheading": "आपले दस्तऐवज संक्षेप करा आणि त्यांना आवाजात ऐका",
        "upload": "दस्तऐवज अपलोड करा (PDF किंवा DOCX)",
        "generate": "⚙️ सारांश तयार करा",
        "read": "🔊 सारांश वाचा",
        "switch_lang": "🔁 भाषा बदला (मराठी/English)",
        "summary_header": "📝 सारांश"
    }
}
# 👇 Spinner messages to encourage healthy habits while summary loads
HEALTH_SPINNER_MESSAGES = [
    "Sit straight dear user 💺", "Give your eyes a blink👀", "I think u should Drink some water 💧",
    "Take a deep breath 🌬️", "Relax your shoulders 🧘",
    "Smile a bit 😊", "Stretch your arms 👐"
]

def show_friendly_spinner(msg):
    spinner_html = f"""
    <div class="custom-spinner">
        <div class="spinner-box">
            <span>{msg}</span>
            <div class="bar-container">
                <div class="bar"></div>
            </div>
        </div>
    </div>
    <style>
    .custom-spinner {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: rgba(0, 0, 0, 0.6); display: flex;
        align-items: center; justify-content: center; z-index: 1000;
        flex-direction: column;
    }}
    .spinner-box {{
        background-color: #222; color: white; border-radius: 12px;
        padding: 30px 40px; box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        font-size: 22px; font-weight: 600;
        font-family: 'Segoe UI Emoji', sans-serif;
        display: flex; align-items: center; gap: 12px; flex-direction: column;
        min-width: 350px;
    }}
    .bar-container {{
        width: 100%; height: 10px; background-color: #444;
        margin-top: 14px; border-radius: 6px; overflow: hidden;
    }}
    .bar {{
        height: 100%; background-color: #4fa1f2;
        animation: loadingbar 4s linear infinite;
    }}
    @keyframes loadingbar {{
        0% {{ width: 0%; }}
        50% {{ width: 80%; }}
        100% {{ width: 0%; }}
    }}
    </style>
    """
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)

def remove_spinner():
    spinner_placeholder.empty()

# [CONTINUED IN NEXT MESSAGE — includes extract_text, summarize_text_distilbart, and UI code...]
from PIL import Image, ImageFilter, ImageOps

from PIL import Image, ImageFilter, ImageOps  # Make sure these are imported at the top

def extract_text(uploaded_file):
    text = ""
    current_pos = uploaded_file.tell()
    uploaded_file.seek(0)

    if uploaded_file.name.endswith(".pdf"):
        pdf_file = io.BytesIO(uploaded_file.read())
        uploaded_file.seek(0)

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text and len(page_text.strip()) > 20:
                    text += page_text + "\n"
                else:
                    # OCR fallback
                    try:
                        image = page.to_image(resolution=300).original
                        gray = image.convert("L")
                        gray = gray.filter(ImageFilter.MedianFilter())
                        gray = gray.point(lambda x: 0 if x < 140 else 255, '1')  # Threshold

                        ocr_text = pytesseract.image_to_string(gray, lang='eng+mar')

                        # Clean gibberish, page markers, and icons
                        ocr_text = re.sub(r'AVV Letters GEN.*?Page.*?\d+', '', ocr_text, flags=re.IGNORECASE)
                        ocr_text = re.sub(r'[•✓✔➡️←→★◆■●►▪☑️🔷]+', '', ocr_text)
                        ocr_text = re.sub(r'[^a-zA-Z0-9.,:/()\-\n ]+', '', ocr_text)
                        ocr_text = re.sub(r'\n{2,}', '\n', ocr_text)
                        ocr_text = ocr_text.strip()

                        if len(ocr_text.strip()) > 50 and any(c.isalpha() for c in ocr_text):
                            text += ocr_text + "\n"

                    except Exception as e:
                        print("OCR error:", e)
                        continue

    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])

    uploaded_file.seek(current_pos)
    return text.strip() if text.strip() else "⚠️ Image-based PDF: No valid text extracted or text is too sparse."

def clean_extracted_text(raw_text):
    raw_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', raw_text)
    raw_text = re.sub(r'\n+', '. ', raw_text)
    raw_text = re.sub(r'\s{2,}', ' ', raw_text)
    return raw_text.strip()

def generate_notice_summary(text):
    lines = text.split('\n')
    summary_lines = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        if 'subject' in line.lower() and len(line) < 150:
            summary_lines.append(f"📌 {line}")
        elif any(k in line.lower() for k in ['issued by', 'executive engineer', 'officer', 'department']):
            summary_lines.append(f"🧾 {line}")
        elif any(k in line.lower() for k in ['last date', 'submission date', 'due date']):
            summary_lines.append(f"🗓️ {line}")
        elif 'portal' in line.lower() or 'website' in line.lower():
            summary_lines.append(f"🌐 {line}")
        elif any(k in line.lower() for k in ['instructions', 'must', 'shall']):
            summary_lines.append(f"📋 {line}")
    return '\n'.join(summary_lines) if len(summary_lines) >= 2 else None

def summarize_text_distilbart(text):
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"Failed to load DistilBart model: {e}")
        return "⚠️ DistilBart model could not be loaded."

    try:
        cleaned_text = text.strip()
        
        # If very little text, return it directly instead of summarizing
        if len(cleaned_text) < 50:
            return f"📸 This seems to be an image-based document with limited content:\n\n{cleaned_text or '⚠️ No readable text found from OCR.'}"

        # Normal summarization process
        chunks = [cleaned_text[i:i+800] for i in range(0, len(cleaned_text), 800)]
        summary = ""
        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue
            try:
                result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                chunk_summary = result[0]['summary_text']
                chunk_summary = chunk_summary.replace(" .", ".").replace(" ,", ",")
                chunk_summary = re.sub(r'\n+', ' ', chunk_summary).strip()
                chunk_summary = chunk_summary[0].upper() + chunk_summary[1:] if chunk_summary else chunk_summary
                summary += chunk_summary + " "
            except:
                continue

        return summary.strip() or "⚠️ No usable summary could be generated."
    except Exception as e:
        return f"⚠️ Summarization error: {e}"

import google.generativeai as genai

# Load Gemini API key from .env or directly as string
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_text_with_gemini(text):
    from google.generativeai import configure, GenerativeModel
    import google.generativeai as genai

    configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = (
        "Summarize the following document in 3-5 bullet points. "
        "The document may contain government notices, civic records, tender information, or scanned OCR text.\n\n"
        f"{text}"
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini summarization failed: {e}"

def describe_tables(uploaded_file):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream", edge_tol=500)
        if not tables or tables.n == 0:
            return None

        return generate_table_summary(tables)
    except Exception as e:
        return f"⚠️ Table summary error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def generate_table_summary(tables):
    garden_count = 0
    found_keywords = {"garden": False, "timing": False, "area": False, "employee": False, "facility": False}
    for table in tables:
        df = table.df
        if df.empty or df.shape[0] < 2:
            continue
        header_text = " ".join(df.iloc[0].astype(str)).lower()
        for key in found_keywords:
            if key in header_text:
                found_keywords[key] = True
        for row in df.iloc[1:].itertuples(index=False):
            first_cell = str(row[0]).strip()
            if first_cell.isdigit():
                garden_count += 1
    if garden_count == 0:
        return None
    lines = ["### 📊 Document Summary:"]
    lines.append(f"- This PDF contains structured data for approximately **{garden_count} records**.")
    details = [k.replace("_", " ") for k, v in found_keywords.items() if v]
    if details:
        lines.append(f"- Each entry includes: {', '.join(details)}.")
    lines.append("- Useful for understanding structured civic or departmental data.")
    return "\n".join(lines)

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return "⚠️ Translation failed."

def speak_text_google(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3')
    except Exception as e:
        st.error(f"Audio failed: {e}")

def speak_text(text, lang='en'):
    is_marathi = any(0x0900 <= ord(char) <= 0x097F for char in text)
    speak_text_google(text, lang='mr' if is_marathi else 'en')

# --- UI ---
lang = st.session_state.language
labels = LABELS[lang]
st.markdown(f"<h1 style='background-color:black; color:white; padding:12px; border-radius:8px; text-align:center;'>{labels['title']}</h1>", unsafe_allow_html=True)
st.caption(labels["subheading"])
uploaded_file = st.file_uploader(labels["upload"], type=["pdf", "docx"])

if uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.summary = ''
    st.session_state.editable_summary = ''
    st.session_state.last_uploaded_file = uploaded_file

if st.checkbox(labels["switch_lang"]):
    st.session_state.language = 'mr' if lang == 'en' else 'en'
    st.rerun()

# ✅ Health spinner messages
HEALTH_SPINNER_MSGS = [
    "💧 Drink water", "💺 Sit straight", "👀 Blink your eyes", "🌬️ Breathe",
    "🧘 Relax your shoulders", "📴 Take a screen break", "✨ Stay focused"
]

if uploaded_file and st.button(labels["generate"]):
    # Show random health spinner instead of fixed message
    show_friendly_spinner(random.choice(HEALTH_SPINNER_MSGS))

    try:
        table_descriptions = None
        if uploaded_file.name.endswith(".pdf"):
            table_descriptions = describe_tables(uploaded_file)
            if table_descriptions and ("⚠️" in table_descriptions or "error" in table_descriptions.lower()):
                table_descriptions = None

        raw_text = extract_text(uploaded_file)
        cleaned_general_text = clean_extracted_text(raw_text)

        important_lines = []
        for line in cleaned_general_text.split('\n'):
            if any(k in line.lower() for k in ['tender fee', 'emd', 'estimated cost', 'submission', 'class of contractor']):
                important_lines.append("- " + line.strip())
        highlighted_info = "\n".join(important_lines)

        custom_summary = generate_notice_summary(cleaned_general_text)

        if custom_summary:
            main_text_summary = custom_summary
        else:
            # First try distilbart, fallback to Gemini if short or warning
            bart_summary = summarize_text_distilbart(cleaned_general_text)
            if "⚠️" in bart_summary or len(bart_summary.strip()) < 50:
                main_text_summary = summarize_text_with_gemini(cleaned_general_text)
            else:
                main_text_summary = bart_summary

        final_content_parts = []
        if table_descriptions:
            final_content_parts.append(table_descriptions)
        if highlighted_info:
            final_content_parts.append("### 📌 Key Financial Info:\n" + highlighted_info)
        final_content_parts.append(main_text_summary)

        combined_content_overview = "\n\n".join(filter(None, [
            *final_content_parts,
            "---" if len(final_content_parts) > 1 else ""
        ])).strip()

        if not combined_content_overview or len(combined_content_overview) < 50:
            combined_content_overview = "⚠️ Summary too short or unclear to translate meaningfully."

        st.session_state.summary_en = combined_content_overview
        st.session_state.summary_mr = translate_text(combined_content_overview, 'mr')
        st.session_state.summary = st.session_state.summary_mr if lang == 'mr' else st.session_state.summary_en
        st.session_state.editable_summary = st.session_state.summary

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    finally:
        remove_spinner()

# ✅ Output Section
if st.session_state.summary:
    st.subheader(labels["summary_header"])
    editable_text = st.text_area("Review and edit the extracted content/summary below:", value=st.session_state.editable_summary, height=200)
    if editable_text != st.session_state.editable_summary:
        st.session_state.editable_summary = editable_text

    target_lang = 'mr' if lang == 'en' else 'en'
    if st.button(f"🌐 Translate Summary to {'Marathi' if target_lang == 'mr' else 'English'}"):
        st.session_state.summary = translate_text(st.session_state.editable_summary, target_lang)
        st.session_state.editable_summary = st.session_state.summary
        st.success("✅ Translated successfully.")
        st.rerun()

    if st.button(labels["read"]):
        show_friendly_spinner("🎧 Generating your voice... Please relax!")
        speak_text(st.session_state.editable_summary.strip(), lang=st.session_state.language)
        remove_spinner()

# ✅ Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Designed and developed by Soha Mujawar</p>", unsafe_allow_html=True)

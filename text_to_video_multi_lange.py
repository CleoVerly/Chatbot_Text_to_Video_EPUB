# app.py

import streamlit as st
import os
import faiss
import numpy as np
import requests
import re
import hashlib
import textwrap
from io import BytesIO

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

# ======================================================================================
# Konfigurasi Multi-Bahasa
# ======================================================================================

translations = {
    "id": {
        "page_title": "Chatbot EPUB",
        "app_title": "🤖 Chatbot Cerdas Berbasis File EPUB",
        "app_subheader": "Pilih buku, ajukan pertanyaan, dan ubah jawabannya menjadi video!",
        "sidebar_title": "📚 Koleksi Buku Anda",
        "sidebar_instruction": "Letakkan file EPUB Anda di dalam folder `epub_files`.",
        "no_epub_files": "Tidak ada file .epub di folder `{epub_dir}`.",
        "select_epub": "Pilih file EPUB:",
        "process_button": "Proses File EPUB Pilihan",
        "processing_epub": "Memproses EPUB: Mengekstrak teks, membuat embedding, dan membangun indeks...",
        "process_success": "Pemrosesan untuk {file_name} selesai!",
        "extract_fail": "Tidak ada teks yang bisa diekstrak dari {file_name}.",
        "no_text_to_process": "Tidak ada teks yang dapat diproses dari file EPUB ini.",
        "active_file": "✅ Aktif: **{file_name}**",
        "start_info": "Pilih file dan klik proses untuk memulai.",
        "chat_placeholder": "Tanyakan sesuatu tentang isi buku ini...",
        "process_first": "Harap proses file EPUB terlebih dahulu.",
        "thinking": "Mencari informasi dan berpikir...",
        "source_expander": "Lihat Sumber Asli dari EPUB",
        "video_button": "🎬 Buat Video dari Jawaban Ini",
        "video_spinner": "Membuat video... Mengunduh gambar dan menyusun {scenes} adegan.",
        "video_success": "Video berhasil dibuat!",
        "video_fail": "Gagal membuat video.",
        "download_video": "Unduh Video",
        "llm_prompt_lang": "Bahasa Indonesia",
        "pexels_key_missing": "Pexels API Key tidak ditemukan. Fitur video tidak akan berfungsi.",
        "openrouter_key_missing": "API Key OpenRouter tidak ditemukan. Harap atur di file .env Anda.",
        "lang_select": "Pilih Bahasa:"
    },
    "en": {
        "page_title": "EPUB Chatbot",
        "app_title": "🤖 Smart EPUB-based Chatbot",
        "app_subheader": "Select a book, ask questions, and turn the answers into videos!",
        "sidebar_title": "📚 Your Book Collection",
        "sidebar_instruction": "Place your EPUB files in the `epub_files` folder.",
        "no_epub_files": "No .epub files found in the `{epub_dir}` folder.",
        "select_epub": "Select an EPUB file:",
        "process_button": "Process Selected EPUB File",
        "processing_epub": "Processing EPUB: Extracting text, creating embeddings, and building index...",
        "process_success": "Processing for {file_name} is complete!",
        "extract_fail": "No text could be extracted from {file_name}.",
        "no_text_to_process": "No processable text found in this EPUB file.",
        "active_file": "✅ Active: **{file_name}**",
        "start_info": "Select a file and click process to begin.",
        "chat_placeholder": "Ask something about the book's content...",
        "process_first": "Please process an EPUB file first.",
        "thinking": "Searching for information and thinking...",
        "source_expander": "View Original Source from EPUB",
        "video_button": "🎬 Create Video from This Answer",
        "video_spinner": "Creating video... Downloading images and composing {scenes} scenes.",
        "video_success": "Video created successfully!",
        "video_fail": "Failed to create video.",
        "download_video": "Download Video",
        "llm_prompt_lang": "English",
        "pexels_key_missing": "Pexels API Key not found. Video feature will be disabled.",
        "openrouter_key_missing": "OpenRouter API Key not found. Please set it in your .env file.",
        "lang_select": "Select Language:"
    },
    "ar": {
        "page_title": "روبوت محادثة EPUB",
        "app_title": "🤖 روبوت محادثة ذكي قائم على ملفات EPUB",
        "app_subheader": "اختر كتابًا، اطرح أسئلة، وحوّل الإجابات إلى فيديوهات!",
        "sidebar_title": "📚 مجموعة كتبك",
        "sidebar_instruction": "ضع ملفات EPUB الخاصة بك في مجلد `epub_files`.",
        "no_epub_files": "لم يتم العثور على ملفات .epub في مجلد `{epub_dir}`.",
        "select_epub": "اختر ملف EPUB:",
        "process_button": "معالجة ملف EPUB المحدد",
        "processing_epub": "جاري معالجة EPUB: استخراج النص، إنشاء التضمينات، وبناء الفهرس...",
        "process_success": "اكتملت معالجة {file_name}!",
        "extract_fail": "تعذر استخراج أي نص من {file_name}.",
        "no_text_to_process": "لا يوجد نص قابل للمعالجة في ملف EPUB هذا.",
        "active_file": "✅ نشط: **{file_name}**",
        "start_info": "اختر ملفًا وانقر على زر المعالجة للبدء.",
        "chat_placeholder": "اسأل شيئًا عن محتوى الكتاب...",
        "process_first": "يرجى معالجة ملف EPUB أولاً.",
        "thinking": "جاري البحث عن المعلومات والتفكير...",
        "source_expander": "عرض المصدر الأصلي من EPUB",
        "video_button": "🎬 إنشاء فيديو من هذه الإجابة",
        "video_spinner": "جاري إنشاء الفيديو... تنزيل الصور وتجميع {scenes} مشاهد.",
        "video_success": "تم إنشاء الفيديو بنجاح!",
        "video_fail": "فشل إنشاء الفيديو.",
        "download_video": "تنزيل الفيديو",
        "llm_prompt_lang": "Arabic",
        "pexels_key_missing": "مفتاح Pexels API غير موجود. ميزة الفيديو ستكون معطلة.",
        "openrouter_key_missing": "مفتاح OpenRouter API غير موجود. يرجى إعداده في ملف .env الخاص بك.",
        "lang_select": "اختر اللغة:"
    }
}

# ======================================================================================
# Konfigurasi dan Pemuatan Model
# ======================================================================================

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if 'lang' not in st.session_state:
    st.session_state.lang = 'id'

LANG = st.session_state.lang
T = translations[LANG]

st.set_page_config(page_title=T["page_title"], layout="wide")

if not OPENROUTER_API_KEY:
    st.error(T["openrouter_key_missing"])
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ... (Sisa fungsi inti tetap sama, hanya teks UI yang akan diubah)
# ======================================================================================
# Fungsi Pemrosesan EPUB (Logika Chatbot)
# ======================================================================================

def extract_text_from_epub(epub_path):
    try:
        book = epub.read_epub(epub_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        full_text = ""
        for item in items:
            content = item.get_body_content()
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                full_text += text + " "
        return full_text
    except Exception as e:
        st.error(f"Gagal membaca file EPUB '{os.path.basename(epub_path)}': {e}")
        return None

@st.cache_resource(show_spinner=T["processing_epub"])
def process_epub(epub_path):
    text = extract_text_from_epub(epub_path)
    if not text:
        st.error(T["extract_fail"].format(file_name=os.path.basename(epub_path)))
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning(T["no_text_to_process"])
        return None, None

    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.success(T["process_success"].format(file_name=os.path.basename(epub_path)))
    return index, chunks

# ======================================================================================
# Fungsi Chatbot (RAG: Retrieval-Augmented Generation)
# ======================================================================================

def find_relevant_chunks(query, index, chunks, model, top_k=10): # Nilai banyaknya top_k digunakan untuk meningkatkan relevansi
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def get_llm_response(query, context, api_key, language_name):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        f"You are an expert AI assistant specializing in analyzing book content. "
        f"Based on the following context, answer the user's question clearly and informatively in {language_name}. "
        f"If the information is not found in the context, state that you cannot find the answer in this book.\n\n"
        f"CONTEXT:\n{' '.join(context)}\n\n"
        f"USER'S QUESTION:\n{query}\n\n"
        "ANSWER:"
    )
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting OpenRouter API: {e}")
        return "Sorry, an error occurred while trying to get an answer."

# ======================================================================================
# Fungsi Pembuatan Video
# ======================================================================================

def text_to_speech_gtts(text, output_path, lang='id'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        st.error(f"Failed to create audio file: {e}")
        return None

def download_image_from_pexels(query, api_key):
    if not api_key: return None
    headers = {"Authorization": api_key}
    try:
        res = requests.get(f"https://api.pexels.com/v1/search?query={query}&per_page=1", headers=headers)
        res.raise_for_status()
        data = res.json()
        if data["photos"]:
            img_url = data["photos"][0]["src"]["landscape"]
            img_data = requests.get(img_url).content
            return Image.open(BytesIO(img_data)).resize((1280, 720))
    except Exception:
        return None
    return None

def create_text_image(text, size=(1280, 720)):
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=60)
    except IOError:
        font = ImageFont.load_default()
    
    wrapped_text = textwrap.fill(text, width=30)
    _, _, text_w, text_h = draw.textbbox((0, 0), wrapped_text, font=font, align="center")
    x = (size[0] - text_w) / 2
    y = (size[1] - text_h) / 2
    
    stroke_width = 2
    for pos in [ (x-stroke_width, y), (x+stroke_width, y), (x, y-stroke_width), (x, y+stroke_width) ]:
        draw.text(pos, wrapped_text, font=font, fill="black", align="center")

    draw.text((x, y), wrapped_text, font=font, fill="white", align="center")
    return img

def generate_video_from_text(text, pexels_api_key, output_path, lang_code):
    audio_path = output_path.replace(".mp4", ".mp3")
    if not text_to_speech_gtts(text, audio_path, lang=lang_code): return None
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    
    text_chunks = textwrap.wrap(text, width=80)
    if not text_chunks: return None
    
    chunk_duration = duration / len(text_chunks)
    video_clips = []
    
    with st.spinner(T["video_spinner"].format(scenes=len(text_chunks))):
        for i, chunk in enumerate(text_chunks):
            keyword_query = "islamic " + " ".join(chunk.split()[:4])
            bg_image = download_image_from_pexels(keyword_query, pexels_api_key)
            if bg_image is None:
                bg_image = Image.new("RGB", (1280, 720), (0, 0, 0))

            background_clip = ImageClip(np.array(bg_image)).set_duration(chunk_duration)
            text_image = create_text_image(chunk)
            text_clip = ImageClip(np.array(text_image)).set_duration(chunk_duration)
            
            composite_clip = CompositeVideoClip([background_clip, text_clip.set_position("center")])
            video_clips.append(composite_clip)

    final_video = concatenate_videoclips(video_clips).set_audio(audio_clip)
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return output_path

# ======================================================================================
# Antarmuka Pengguna (UI) Streamlit
# ======================================================================================

st.title(T["app_title"])
st.markdown(T["app_subheader"])

# --- SIDEBAR ---
with st.sidebar:
    st.title(T["sidebar_title"])

    lang_options = {"Indonesia": "id", "English": "en", "العربية": "ar"}
    
    def on_lang_change():
        st.session_state.lang = lang_options[st.session_state.selectbox_lang]

    st.selectbox(
        T["lang_select"],
        options=lang_options.keys(),
        key="selectbox_lang",
        on_change=on_lang_change
    )

    st.markdown(T["sidebar_instruction"])
    epub_dir = "epub_files"
    if not os.path.exists(epub_dir): os.makedirs(epub_dir)
    epub_files = [f for f in os.listdir(epub_dir) if f.endswith('.epub')]

    if not epub_files:
        st.warning(T["no_epub_files"].format(epub_dir=epub_dir))
        st.stop()

    selected_epub = st.selectbox(T["select_epub"], epub_files)

    if st.button(T["process_button"], type="primary"):
        epub_path = os.path.join(epub_dir, selected_epub)
        index, chunks = process_epub(epub_path)
        if index is not None and chunks is not None:
            st.session_state.processed_data = {"file_name": selected_epub, "index": index, "chunks": chunks}
            st.session_state.messages = []
            st.rerun()

    if 'processed_data' in st.session_state:
        st.success(T["active_file"].format(file_name=st.session_state.processed_data['file_name']))
    else:
        st.info(T["start_info"])
    
    if not PEXELS_API_KEY:
        st.warning(T["pexels_key_missing"])


# --- AREA CHAT UTAMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "context" in message and message["context"]:
                with st.expander(T["source_expander"]):
                    st.markdown(f"_{'---'.join(message['context'])}_")
            
            video_placeholder = st.empty()
            if "video_path" in message and os.path.exists(message["video_path"]):
                with video_placeholder.container():
                    st.video(message["video_path"])
                    with open(message["video_path"], "rb") as f:
                        st.download_button(T["download_video"], f, file_name=os.path.basename(message["video_path"]))
            elif PEXELS_API_KEY:
                if st.button(T["video_button"], key=f"vid_{i}"):
                    with video_placeholder.container():
                        output_dir = "output_videos"
                        if not os.path.exists(output_dir): os.makedirs(output_dir)
                        file_hash = hashlib.md5(message["content"].encode()).hexdigest()
                        video_path = os.path.join(output_dir, f"video_{file_hash}_{LANG}.mp4")
                        
                        generated_path = generate_video_from_text(message["content"], PEXELS_API_KEY, video_path, LANG)
                        
                        if generated_path:
                            st.session_state.messages[i]["video_path"] = generated_path
                            st.rerun()
                        else:
                            st.error(T["video_fail"])

if prompt := st.chat_input(T["chat_placeholder"]):
    if 'processed_data' not in st.session_state:
        st.warning(T["process_first"])
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    model = load_embedding_model()
    index = st.session_state.processed_data['index']
    chunks = st.session_state.processed_data['chunks']

    with st.chat_message("assistant"):
        with st.spinner(T["thinking"]):
            # PERUBAHAN: Memanggil dengan nilai top_k yang lebih tinggi secara eksplisit
            relevant_context = find_relevant_chunks(prompt, index, chunks, model, top_k=7)
            response = get_llm_response(prompt, relevant_context, OPENROUTER_API_KEY, T["llm_prompt_lang"])
            
            st.markdown(response)
            with st.expander(T["source_expander"]):
                st.markdown(f"_{'---'.join(relevant_context)}_")
            
            st.session_state.messages.append({"role": "assistant", "content": response, "context": relevant_context})
            st.rerun()

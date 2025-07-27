# app_fixed.py

import streamlit as st
import os
import faiss
import numpy as np
import requests
import re
import hashlib
import textwrap
import json
from io import BytesIO

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

# ======================================================================================
# Konfigurasi Multi-Bahasa
# ======================================================================================

translations = {
    "id": {
        "page_title": "Chatbot EPUB",
        "app_title": "🤖 Chatbot Cerdas Berbasis File EPUB",
        "app_subheader": "Unggah & pilih buku, ajukan pertanyaan, dan ubah jawabannya menjadi video!",
        "sidebar_title": "📚 Koleksi & Pengaturan",
        "lang_select": "Pilih Bahasa:",
        "upload_title": "Unggah File EPUB Baru",
        "upload_label": "Pilih file .epub untuk diunggah",
        "upload_success": "File '{filename}' berhasil diunggah!",
        "manage_files_title": "Kelola File EPUB",
        "select_to_delete": "Pilih file untuk dihapus:",
        "delete_button": "Hapus File Pilihan",
        "delete_success": "File '{filename}' berhasil dihapus.",
        "delete_fail": "Gagal menghapus file: {error}",
        "select_epub": "Pilih file EPUB untuk diproses:",
        "process_button": "Proses File Pilihan",
        "processing_epub": "Memproses EPUB: Mengekstrak teks, membuat embedding, dan membangun indeks...",
        "process_success": "Pemrosesan untuk {file_name} selesai!",
        "active_file": "✅ Aktif: **{file_name}**",
        "start_info": "Pilih file dan klik proses untuk memulai.",
        "chat_placeholder": "Tanyakan sesuatu tentang isi buku ini...",
        "process_first": "Harap proses file EPUB terlebih dahulu.",
        "thinking": "Mencari & memeringkat ulang informasi...", 
        "source_expander": "Lihat Sumber Asli dari EPUB",
        "video_button": "🎬 Buat Video dari Jawaban Ini",
        "video_spinner": "Membuat video... Mengunduh gambar dan menyusun {scenes} adegan.",
        "video_fail": "Gagal membuat video.",
        "download_video": "Unduh Video",
        "llm_prompt_lang": "Bahasa Indonesia"
    },
    "en": {
        "page_title": "EPUB Chatbot",
        "app_title": "🤖 Smart EPUB-based Chatbot",
        "app_subheader": "Upload & select a book, ask questions, and turn answers into videos!",
        "sidebar_title": "📚 Collection & Settings",
        "lang_select": "Select Language:",
        "upload_title": "Upload New EPUB File",
        "upload_label": "Choose a .epub file to upload",
        "upload_success": "File '{filename}' uploaded successfully!",
        "manage_files_title": "Manage EPUB Files",
        "select_to_delete": "Select a file to delete:",
        "delete_button": "Delete Selected File",
        "delete_success": "File '{filename}' has been deleted.",
        "delete_fail": "Failed to delete file: {error}",
        "select_epub": "Select an EPUB file to process:",
        "process_button": "Process Selected File",
        "processing_epub": "Processing EPUB: Extracting text, creating embeddings, and building index...",
        "process_success": "Processing for {file_name} is complete!",
        "active_file": "✅ Active: **{file_name}**",
        "start_info": "Select a file and click process to begin.",
        "chat_placeholder": "Ask something about the book's content...",
        "process_first": "Please process an EPUB file first.",
        "thinking": "Searching & re-ranking information...", 
        "source_expander": "View Original Source from EPUB",
        "video_button": "🎬 Create Video from This Answer",
        "video_spinner": "Creating video... Downloading images and composing {scenes} scenes.",
        "video_fail": "Failed to create video.",
        "download_video": "Download Video",
        "llm_prompt_lang": "English"
    },
    "ar": {
        "page_title": "روبوت محادثة EPUB",
        "app_title": "🤖 روبوت محادثة ذكي قائم على ملفات EPUB",
        "app_subheader": "قم بتحميل واختيار كتاب، اطرح أسئلة، وحوّل الإجابات إلى فيديوهات!",
        "sidebar_title": "📚 المجموعة والإعدادات",
        "lang_select": "اختر اللغة:",
        "upload_title": "تحميل ملف EPUB جديد",
        "upload_label": "اختر ملف .epub للتحميل",
        "upload_success": "تم تحميل الملف '{filename}' بنجاح!",
        "manage_files_title": "إدارة ملفات EPUB",
        "select_to_delete": "اختر ملفًا لحذفه:",
        "delete_button": "حذف الملف المحدد",
        "delete_success": "تم حذف الملف '{filename}'.",
        "delete_fail": "فشل حذف الملف: {error}",
        "select_epub": "اختر ملف EPUB للمعالجة:",
        "process_button": "معالجة الملف المحدد",
        "processing_epub": "جاري معالجة EPUB: استخراج النص، إنشاء التضمينات، وبناء الفهرس...",
        "process_success": "اكتملت معالجة {file_name}!",
        "active_file": "✅ نشط: **{file_name}**",
        "start_info": "اختر ملفًا وانقر على زر المعالجة للبدء.",
        "chat_placeholder": "اسأل شيئًا عن محتوى الكتاب...",
        "process_first": "يرجى معالجة ملف EPUB أولاً.",
        "thinking": "البحث وإعادة ترتيب المعلومات...", 
        "source_expander": "عرض المصدر الأصلي من EPUB",
        "video_button": "🎬 إنشاء فيديو من هذه الإجابة",
        "video_spinner": "جاري إنشاء الفيديو... تنزيل الصور وتجميع {scenes} مشاهد.",
        "video_fail": "فشل إنشاء الفيديو.",
        "download_video": "تنزيل الفيديو",
        "llm_prompt_lang": "Arabic"
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
    st.error("OpenRouter API Key not found. Please set it in your .env file.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ======================================================================================
# Fungsi Inti
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
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        return None, None

    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.success(T["process_success"].format(file_name=os.path.basename(epub_path)))
    return index, chunks

def is_summary_question(question):
    """Mendeteksi apakah pertanyaan bersifat umum/ringkasan."""
    summary_keywords = ["apa isi buku", "tentang apa", "ringkasan", "rangkuman", "summarize", "buku ini membahas"]
    return any(keyword in question.lower() for keyword in summary_keywords)

def find_relevant_chunks(query, index, chunks, model, top_k=20): # PERBAIKAN: Ambil lebih banyak kandidat
    """Mencari potongan teks yang relevan menggunakan query asli pengguna."""
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def rerank_chunks_with_llm(query, chunks_to_rerank, api_key):
    """Menggunakan LLM cepat untuk memeringkat ulang daftar chunk."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    chunk_list_str = ""
    for i, chunk in enumerate(chunks_to_rerank):
        chunk_list_str += f"CHUNK {i+1}:\n\"{chunk}\"\n\n"

    prompt = (
        "You are a relevance ranking assistant. From the list of chunks below, identify the numbers of the TOP 7 most relevant chunks for answering the user's question. "
        "Return your answer ONLY as a JSON list of numbers. Example: [1, 5, 3, 10, 2, 8, 12]\n\n"
        f"USER'S QUESTION:\n\"{query}\"\n\n"
        f"LIST OF CHUNKS:\n{chunk_list_str}\n\n"
        "JSON list of top 7 chunk numbers:"
    )
    
    data = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        ranked_indices = json.loads(content.strip())
        # Konversi indeks 1-based dari LLM ke 0-based
        reranked_chunks = [chunks_to_rerank[i-1] for i in ranked_indices if 0 < i <= len(chunks_to_rerank)]
        return reranked_chunks
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Re-ranking gagal: {e}. Kembali ke metode awal.")
        return chunks_to_rerank[:7] # Fallback jika gagal

def get_llm_response(query, context, api_key, language_name, is_summary=False):
    """Menghasilkan jawaban dari LLM berdasarkan konteks dan niat."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    if is_summary:
        instruction = "Based on the following sample of text chunks from a book, provide a general summary of the main topics discussed."
    else:
        instruction = "Based on the context, answer the user's original question."

    prompt = (
        f"You are an expert AI assistant. {instruction} "
        f"Answer in {language_name}. If the info isn't in the context, say you cannot find the answer in the book.\n\n"
        f"CONTEXT:\n{' '.join(context)}\n\nUSER'S QUESTION:\n{query}\n\nANSWER:"
    )
    
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# ... (Fungsi untuk video tetap sama)
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
            if bg_image is None: bg_image = Image.new("RGB", (1280, 720), (0, 0, 0))

            background_clip = ImageClip(np.array(bg_image)).with_duration(chunk_duration)
            text_image = create_text_image(chunk)
            text_clip = ImageClip(np.array(text_image)).with_duration(chunk_duration)
            
            composite_clip = CompositeVideoClip([background_clip, text_clip.with_position("center")])
            video_clips.append(composite_clip)

    final_video = concatenate_videoclips(video_clips).with_audio(audio_clip)
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    
    if os.path.exists(audio_path): os.remove(audio_path)
    return output_path

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
    for pos in [ (x-stroke_width, y), (x+stroke_width, y), (x, y-stroke_width), (y+stroke_width, y) ]:
        draw.text(pos, wrapped_text, font=font, fill="black", align="center")

    draw.text((x, y), wrapped_text, font=font, fill="white", align="center")
    return img

def text_to_speech_gtts(text, output_path, lang='id'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
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

# ======================================================================================
# Antarmuka Pengguna (UI) Streamlit
# ======================================================================================

st.title(T["app_title"])
st.markdown(T["app_subheader"])

# --- SIDEBAR ---
with st.sidebar:
    st.title(T["sidebar_title"])

    lang_map = {"Indonesia": "id", "English": "en", "العربية": "ar"}
    lang_display_names = list(lang_map.keys())
    lang_codes = list(lang_map.values())
    
    current_lang_index = lang_codes.index(st.session_state.lang)

    def on_lang_change():
        selected_display_name = st.session_state.selectbox_lang
        st.session_state.lang = lang_map[selected_display_name]

    st.selectbox(
        T["lang_select"],
        options=lang_display_names,
        index=current_lang_index,
        key="selectbox_lang",
        on_change=on_lang_change
    )
    st.divider()

    with st.expander(T["upload_title"]):
        uploaded_file = st.file_uploader(T["upload_label"], type=['epub'])
        if uploaded_file is not None:
            epub_dir = "epub_files"
            if not os.path.exists(epub_dir):
                os.makedirs(epub_dir)
            
            file_path = os.path.join(epub_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(T["upload_success"].format(filename=uploaded_file.name))
            st.rerun()

    epub_dir = "epub_files"
    if not os.path.exists(epub_dir): os.makedirs(epub_dir)
    epub_files = [f for f in os.listdir(epub_dir) if f.endswith('.epub')]

    with st.expander(T["manage_files_title"]):
        if not epub_files:
            st.info("No EPUB files to manage.")
        else:
            file_to_delete = st.selectbox(T["select_to_delete"], epub_files, index=None, placeholder="Select a file...")
            if file_to_delete:
                if st.button(T["delete_button"], type="primary"):
                    try:
                        file_path = os.path.join(epub_dir, file_to_delete)
                        os.remove(file_path)
                        if 'processed_data' in st.session_state and st.session_state.processed_data['file_name'] == file_to_delete:
                            del st.session_state['processed_data']
                        st.success(T["delete_success"].format(filename=file_to_delete))
                        st.rerun()
                    except Exception as e:
                        st.error(T["delete_fail"].format(error=e))
    
    st.divider()

    if not epub_files:
        st.warning("Please upload an EPUB file to begin.")
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
            # PERBAIKAN: Logika adaptif dengan re-ranking
            summary_flag = is_summary_question(prompt)
            if summary_flag:
                relevant_context = chunks[:15]
            else:
                # 1. Ambil lebih banyak kandidat
                initial_chunks = find_relevant_chunks(prompt, index, chunks, model, top_k=20)
                # 2. Peringkat ulang untuk mendapatkan yang terbaik
                relevant_context = rerank_chunks_with_llm(prompt, initial_chunks, OPENROUTER_API_KEY)
            
            response = get_llm_response(prompt, relevant_context, OPENROUTER_API_KEY, T["llm_prompt_lang"], is_summary=summary_flag)
            
            st.markdown(response)
            with st.expander(T["source_expander"]):
                st.markdown(f"_{'---'.join(relevant_context)}_")
            
            st.session_state.messages.append({"role": "assistant", "content": response, "context": relevant_context})
            st.rerun()

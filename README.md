
# 📚 Text-to-Video Generator dari ePub

Aplikasi ini mengubah isi file **ePub** menjadi video dengan **suara narasi otomatis (TTS)** dan **teks berjalan** menggunakan Python dan Streamlit.

## 🚀 Fitur Utama

- 📖 Mendukung input file `.epub`
- 🧠 Menggunakan *SentenceTransformer* untuk pemrosesan semantik
- 🔊 Menggunakan **gTTS** untuk konversi teks ke suara
- 🎬 Membuat video dari teks + suara menggunakan **MoviePy**
- 💬 Antarmuka interaktif berbasis **Streamlit**

## 🧰 Teknologi yang Digunakan

- Python
- Streamlit
- gTTS
- MoviePy
- SentenceTransformer
- BeautifulSoup + ebooklib (untuk ekstrak konten ePub)
- LangChain Text Splitter
- PIL (Python Imaging Library)

## 🛠️ Instalasi

1. Clone repo ini dan masuk ke direktori proyek:

```bash
git clone <url-repo>
cd project_chatbot_llm
```

2. Buat dan aktifkan environment (opsional):

```bash
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows
```

3. Install semua dependensi:

```bash
pip install -r requirements.txt
```

4. Buat file `.env` untuk menyimpan API Key:

Buat file bernama `.env` di root folder dan isi dengan format berikut:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
PEXELS_API_KEY=your_pexels_api_key
```

> Gantilah nilai `your_openrouter_api_key` dan `your_pexels_api_key` dengan API key milikmu.

## ▶️ Cara Menjalankan

```bash
streamlit run text_to_video.py
```

## 📂 Struktur Proyek (Ringkasan)

```
project_chatbot_llm/
│
├── text_to_video.py       # Aplikasi Streamlit untuk konversi teks ke video
├── text_to_text.py        # File tambahan (tidak dijelaskan)
├── .env                   # Variabel lingkungan
└── .git/                  # Repo Git lokal
```

## 📸 Contoh Output

Video dengan teks dan suara narasi otomatis.  
*Tambahkan cuplikan video atau gambar jika ada.*

## 📝 Lisensi

[MIT License](LICENSE) – Silakan gunakan, modifikasi, dan distribusikan dengan bebas.

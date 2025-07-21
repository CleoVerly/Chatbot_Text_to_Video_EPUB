# 🤖 Chatbot EPUB Cerdas dengan Konversi Video Multi-Bahasa

Aplikasi ini adalah sebuah chatbot cerdas berbasis RAG (Retrieval-Augmented Generation) yang memungkinkan pengguna untuk berinteraksi dengan konten dari file `.epub`. Pengguna dapat mengunggah buku, mengajukan pertanyaan dalam berbagai bahasa, dan mengubah jawaban yang dihasilkan oleh AI menjadi sebuah video pendek lengkap dengan narasi suara dan visual yang relevan.

Proyek ini juga dilengkapi dengan *suite* evaluasi untuk mengukur kualitas dan akurasi dari pipeline RAG yang digunakan.

## ✨ Fitur Utama

- **🖥️ Antarmuka Interaktif**: Dibangun dengan Streamlit untuk kemudahan penggunaan.
- **📚 Manajemen EPUB Dinamis**: Unggah dan hapus file `.epub` langsung dari antarmuka tanpa perlu merestart aplikasi.
- **🌍 Dukungan Multi-Bahasa**: Antarmuka dan output (teks & suara) tersedia dalam Bahasa Indonesia (id), Inggris (en), dan Arab (ar).
- **🧠 Pipeline RAG Canggih**:
    - **Retrieval Awal**: Menggunakan `FAISS` dan `SentenceTransformer` untuk pencarian kemiripan semantik yang cepat.
    - **LLM Re-Ranking**: Memanfaatkan model LLM yang ringan (`Mistral-7B`) untuk menyusun ulang dan memilih konteks yang paling relevan dari hasil pencarian awal, sehingga meningkatkan akurasi.
    - **Generasi Jawaban**: Menggunakan model LLM canggih (`GPT-4o Mini`) via OpenRouter untuk menghasilkan jawaban yang natural dan faktual berdasarkan konteks.
- **🎬 Konversi Teks-ke-Video Otomatis**:
    - **Narasi Suara (TTS)**: Menggunakan Google Text-to-Speech (gTTS) untuk menghasilkan audio dalam bahasa yang dipilih.
    - **Visual Dinamis**: Mencari dan mengunduh gambar latar belakang yang relevan dari Pexels API untuk setiap adegan video.
    - **Komposisi Video**: Menggabungkan gambar, teks overlay, dan audio menjadi file video `.mp4` menggunakan MoviePy.
- **🧪 Suite Evaluasi Bawaan**: Termasuk skrip `evaluate.py` untuk mengukur performa sistem RAG menggunakan metrik seperti *Retrieval Accuracy*, *Faithfulness*, *F1-Score*, dan *Semantic Similarity*.

## 📂 Struktur Proyek

```
CHATBOT_TEXT_TO_VIDEO_EPUB/
│
├── app.py                      # File utama aplikasi Streamlit (versi terbaru)
├── base_app/
│   ├── text_to_text.py         # Versi awal chatbot (hanya teks)
│   ├── text_to_video.py        # Versi awal konverter teks-ke-video
│   └── text_to_video_multi_lange.py # Versi pengembangan dengan multi-bahasa
│
├── epub_files/                 # Tempat menyimpan file .epub yang diunggah
│   └── (contoh: book.epub)
│
├── evaluation/
│   ├── evaluate.py             # Skrip untuk menjalankan evaluasi otomatis
│   └── golden_dataset.json     # Dataset untuk evaluasi (pertanyaan & jawaban ideal)
│
├── output_videos/              # Tempat menyimpan hasil video yang telah dibuat
│   └── (contoh: video_xyz.mp4)
│
├── temp_images/                # (Direktori sementara untuk cache gambar, jika diperlukan)
│
├── .env                        # File untuk menyimpan API keys (JANGAN di-commit ke Git)
├── .gitignore                  # Mengabaikan file dan direktori tertentu dari Git
└── README.md                   # Dokumentasi proyek (file ini)
```

## 🛠️ Instalasi & Konfigurasi

Pastikan Anda memiliki Python 3.8+ terinstal di sistem Anda.

1.  **Clone Repositori**

    ```bash
    git clone <url-repositori-anda>
    cd CHATBOT_TEXT_TO_VIDEO_EPUB
    ```

2.  **Buat Virtual Environment** (direkomendasikan)

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    .\venv\Scripts\activate   # Untuk Windows
    ```

3.  **Install Dependensi**

    Buat file `requirements.txt` yang berisi semua library yang dibutuhkan, lalu jalankan:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan API Keys**

    Buat file bernama `.env` di direktori utama proyek dan isikan dengan API key Anda:

    ```env
    OPENROUTER_API_KEY="sk-or-v1-..."
    PEXELS_API_KEY="your_pexels_api_key_here"
    ```

    - `OPENROUTER_API_KEY`: Diperlukan untuk mengakses model LLM (Mistral & GPT-4o Mini).
    - `PEXELS_API_KEY`: Diperlukan untuk fitur pembuatan video (mengunduh gambar latar).

5.  **Siapkan Direktori**

    Pastikan direktori `epub_files` dan `output_videos` ada di dalam proyek Anda.

## ▶️ Cara Menjalankan

### 1. Menjalankan Aplikasi Utama

Pastikan Anda berada di direktori utama proyek, lalu jalankan perintah berikut di terminal:

```bash
streamlit run app.py
```

Buka browser Anda dan arahkan ke alamat URL lokal yang ditampilkan (biasanya `http://localhost:8501`).

**Alur Penggunaan:**

1. Pilih bahasa antarmuka di sidebar.
2. Unggah file `.epub` melalui menu di sidebar.
3. Pilih file yang ingin diajak "bicara" dari dropdown.
4. Klik tombol **"Proses File Pilihan"** dan tunggu hingga selesai.
5. Ketik pertanyaan Anda di kolom chat dan tekan Enter.
6. Setelah jawaban muncul, klik tombol **"🎬 Buat Video dari Jawaban Ini"** untuk memulai konversi ke video.

### 2. Menjalankan Skrip Evaluasi

Untuk mengukur performa sistem, Anda dapat menjalankan skrip evaluasi. Pastikan `golden_dataset.json` sudah terisi dan file-file EPUB yang dirujuk ada di dalam folder `epub_files`.

```bash
python evaluation/evaluate.py
```

Skrip akan memproses setiap pertanyaan dalam dataset, menghasilkan jawaban, dan membandingkannya dengan *ground truth* untuk menghasilkan laporan akurasi.

## 🧰 Teknologi yang Digunakan

- **Framework Aplikasi**: Streamlit
- **Pemrosesan Bahasa (NLP)**:
    - `Sentence-Transformers`: Untuk membuat embedding teks.
    - `LangChain`: Untuk text splitting.
    - `OpenRouter.ai`: Sebagai gateway ke model LLM (Mistral, GPT-4o Mini).
- **Pencarian Vektor**: `FAISS` (Facebook AI Similarity Search)
- **Ekstraksi Konten**: `EbookLib` & `BeautifulSoup4`
- **Generasi Multimedia**:
    - `gTTS` (Google Text-to-Speech): Untuk audio.
    - `MoviePy`: Untuk kompilasi video.
    - `Pillow (PIL)`: Untuk manipulasi gambar.
- **API & Lainnya**: `Requests`, `python-dotenv`

## 📝 Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](https://www.google.com/search?q=LICENSE).

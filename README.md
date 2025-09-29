# HealixAI

HealixAI is an AI-powered healthcare assistant that detects skin conditions from images and provides personalized medical advice.  
It combines **deep learning**, **speech-to-text transcription**, and **large language models (LLMs)** to guide patients in understanding their condition and possible next steps.

---

## 🚀 Features

- **Skin Condition Detection**  
  Upload a skin image and receive a model-generated prediction of possible conditions.

- **Symptom Input**  
  Provide additional symptoms via:
  - **Text** input, or
  - **Voice** input (powered by [AssemblyAI](https://www.assemblyai.com/) for speech-to-text transcription).

- **AI-Driven Advice**  
  An integrated LLM processes the detected condition and user symptoms to generate relevant guidance and possible next steps.

- **Interactive Frontend**  
  A simple, responsive interface for image upload, symptom entry, and real-time feedback.

---

## 🧠 How It Works

1. **Image Classification**  
   - HealixAI uses a **pretrained deep learning model** (fine-tuned on dermatological datasets) to classify skin conditions.  
   - Model weights are stored in `final_model_weights.pth` and metadata in `final_model_meta.json`.

2. **Symptom Collection**  
   - Users describe their symptoms through text or record a voice note.  
   - Voice inputs are transcribed to text using AssemblyAI.

3. **LLM Integration**  
   - The predicted condition and user symptoms are passed to a large language model.  
   - The LLM generates an easy-to-understand explanation and practical next steps.

---

## 🏗️ Tech Stack

| Layer       | Technology                  |
|-------------|------------------------------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Frontend**| React + Vite (or Next.js if used) |
| **Model**   | PyTorch (for skin detection) |
| **Voice API** | AssemblyAI |
| **AI Advice** | Groq API (gpt-oss-20b) |
| **Database (optional)** | SQLite / ChromaDB for embeddings |

---

## 📂 Project Structure
```
HealixAI/
├── backend/
│ ├── app/
│ │ ├── api/ # FastAPI endpoints
│ │ ├── models/ # ML model loading and prediction
│ │ ├── services/ # Voice service, LLM integration
│ │ └── main.py # FastAPI entry point
│ └── requirements.txt # Backend dependencies
│
├── frontend/
│ ├── src/ # React/Vite components
│ ├── package.json # Frontend dependencies
│ └── ...
│
├── models/
│ ├── final_model_weights.pth # Trained model weights
│ └── final_model_meta.json # Model metadata
│
└── README.md


---
```
## ⚡ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/HealixAI.git
cd HealixAI
```
### 2️⃣ Backend Setup
cd backend
python -m venv healix
source healix/bin/activate      # Linux/Mac
healix\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

```
Create a .env file inside backend/:
```
ASSEMBLYAI_API_KEY=your_assemblyai_key
OPENAI_API_KEY=your_openai_key
```
```
Run the FastAPI server:
```
uvicorn app.main:app --reload
```

The API will be available at:
👉 http://127.0.0.1:8000

### 3️⃣ Frontend Setup
```
cd ../frontend
npm install
npm run dev
```

The frontend will be available at:
👉 http://localhost:5173
 (default for Vite)

🧪 Model Details

Architecture: A pretrained CNN backbone (e.g., ResNet, EfficientNet) fine-tuned for skin condition detection.

Input: JPG/PNG skin images.

Output: Predicted condition label + confidence score.

Files:

final_model_weights.pth – PyTorch model weights.

final_model_meta.json – Model class mappings and configuration.

⚠️ Note: These files are intentionally tracked in Git despite global .gitignore rules.

🗂️ Environment Variables
Variable	Purpose
ASSEMBLYAI_API_KEY	API key for AssemblyAI transcription
OPENAI_API_KEY	API key for the LLM provider

Store these in .env and never commit them to GitHub.

🗃️ Data and Database

chromadb.sqlite3 (if used) stores vector embeddings for conversation memory.

This file is not included in the repo because it can be rebuilt from your documents and may grow large.

🧑‍⚕️ Usage

Open the web interface.

Upload a clear photo of the affected skin area.

Enter or speak your symptoms.

Review:

Risk Assessment: AI model prediction with confidence.

AI Advice: Suggested next steps or possible treatments.

⚠️ Disclaimer

HealixAI is not a substitute for professional medical advice.
Always consult a qualified healthcare provider for diagnosis or treatment decisions.

🤝 Contributing

Contributions are welcome!

Open an issue for bugs or feature requests.

Submit a pull request with a clear description.

📜 License

This project is licensed under the MIT License
.
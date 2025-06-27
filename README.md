
# PCB Inspection System

This project is a full-stack AI-powered PCB defect detection system using a FastAPI backend and a Vite (React) frontend.

---

## 🔧 Backend (FastAPI)

### ▶ Requirements
- Python 3.8+
- pip
- virtualenv (recommended)

### 📍 Navigate to backend folder
```bash
cd pcb_inspection-main/pcb_inspection-main/backend
```

### 🧪 Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
# OR
source venv/bin/activate   # macOS/Linux
```

### 📦 Install Python dependencies
```bash
pip install -r requirements.txt
```

### ▶ Run FastAPI server
```bash
uvicorn main:app --reload
```

Backend will be running at:
- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

---

## 🌐 Frontend (Vite + React)

### ▶ Requirements
- Node.js (v16 or later recommended)
- npm (comes with Node.js)

### 📍 Navigate to frontend folder
```bash
cd ../frontend
```

### 📦 Install Node dependencies
```bash
npm install
```

### ▶ Run React Dev Server
```bash
npm run dev
```

Frontend will be running at:
- http://127.0.0.1:5173

---

## 🔄 Workflow

1. Upload a PCB image via the frontend.
2. The image is sent to the FastAPI `/predict` endpoint.
3. The backend returns an annotated image with defect classification.
4. The image is displayed or downloaded in the frontend.

---

## 📁 Project Structure

```
pcb_inspection-main/
├── backend/        # FastAPI backend
│   ├── main.py
│   ├── predictor.py
│   ├── ...
├── frontend/       # React frontend
│   ├── src/
│   ├── index.html
│   ├── ...
└── README.md
```

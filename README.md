HEAD
# AI-Powered Triage System 🌡️

## 🎉 Overview
The **_AITriage System_** is a 🔥 cutting-edge tool designed to boost heathcare services efficiency during emergencies in hospitals,this project focuses on assigning triage during during the scorching summer season in India when there are alot of cases in hospitals with summer specific symptoms 🚑 This system harnesses the power of artificial intelligence to predict patient triage levels (🌱 Green, 🟡 Yellow, 🔴 Red) based on vital signs (age, blood pressure, heart rate, temperature, respiratory rate, cholesterol) and symptoms. With multi-language support for 🌐 English and 🇮🇳 Hindi, it caters to diverse healthcare pros and patients. Built with a sleek Streamlit frontend and a robust FastAPI backend, it logs data to a SQLite database. 💾 

## ✨ Features
- 🤖 AI-driven triage prediction using a pre-trained TensorFlow model.
- 🌐 Multi-language interface (English and Hindi).
- 🎨 Real-time vital sign input via an intuitive Streamlit UI.
- ⚡ FastAPI backend for processing predictions and logging.
- 📩 Simulated "Send to Doctor" functionality to display patient details.
- 💾 SQLite database integration for record-keeping .  

## Dataset 
Here we used a heart disease dataset intially then we tweaked it accoring to Inida summer specific data (this data can be any based of the conditions ) and added vitals for different symptoms and then assigned triage to it

## 🛠️ Prerequisites
- 🐍 **Python 3.8 or higher**
- 📦 Git (for cloning the repository)
- 📋 Required Python packages (in `requirements.txt`)
- 🧠 Pre-trained model file (`triage_model_summer.h5`) and scaler (`scaler_summer.pkl`)
- ✍️ A code editor (e.g., VS Code, PyCharm) or terminal access 


## 🛠️ Tech Stack

| Technology  | Usage |
|-------------|-------|
| **FastAPI** | Backend API for prediction |
| **Streamlit** | Frontend interface |
| **Scikit-learn / TensorFlow / PyTorch** | ML model (based on your implementation) |
| **Pydantic** | Data validation |
| **Uvicorn** | ASGI server for FastAPI |
| **Pandas & NumPy** | Data preprocessing |
| **SQLite | DataBase |
---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ai-triage-system.git
cd ai-triage-system
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---


## 🚀 How to Run

### 1. Start the FastAPI backend

```bash
cd backend
uvicorn main:app --reload
```

### 2. Start the Streamlit frontend

In a new terminal:
```bash
cd frontend
streamlit run app.py
```
---

## 📌 Future Improvements

- Add authentication for different roles (doctor, nurse, admin)  
- Integrate with hospital management systems  
- Support for more languages and Voice Assistance
- Deploy via Docker / cloud (e.g., Azure, AWS, GCP)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

Let me know if you'd like to include model training steps, add badges, or want a version with dark-mode themes for Streamlit!


193a815 (Your commit message)

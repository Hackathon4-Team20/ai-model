# 🤖 AI-Based Customer Satisfaction Tracker

This project is a smart tool that evaluates and tracks **customer satisfaction** during conversations using LLMs from [OpenRouter.ai](https://openrouter.ai). It provides:

- A web app interface built with **Streamlit**
- A backend tracking engine using **Python + LLM**
- Real-time score updates for user messages

---

## 🚀 Features

- Tracks customer satisfaction (scale: 1 to 5)
- Ignores greetings or empty messages
- Uses LLM to interpret tone/meaning
- Interactive UI for chatting and feedback
- Built-in satisfaction scoring logic

---

## 📁 Project Structure

```
.
├── app.py                    # Main Streamlit app interface
├── satisfaction_tracker.py   # Satisfaction tracker class
├── server.py                 # (Optional) server logic
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 🔧 Setup Instructions

### 1. 🔁 Clone the Repository

```bash
git clone https://github.com/Hackathon4-Team20/ai-model.git
cd ai-model
```

*Alternatively:* Download the ZIP file and extract it.

### 2. 🐍 Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔐 API Key Setup

You'll need an API key from OpenRouter:
1. When the app starts, it will prompt you for the API key
2. Paste your key in the input field (never commit this to the repository)

### 5. ▶️ Running the Application

Launch the server with:

```bash
python server.py
```

---

## 🧠 How It Works

1. You chat via the web app interface
2. Your messages are analyzed using OpenRouter's LLM
3. A satisfaction score (1–5) is calculated and displayed in real-time

---

---

## ⚠️ Important Notes

* Never commit your API key to the repository
* For production use, consider:
   * Using environment variables (`.env` file)
   * Implementing proper security measures
   * Rate limiting your API calls

---

## 💡 Additional Options (Available on Request)

I can help you add:
* `.env` setup examples
* Docker support
* Hosting instructions (Streamlit Cloud, etc.)
* Collaboration guidelines

Let me know if you'd like to enhance this setup for deployment or team collaboration!

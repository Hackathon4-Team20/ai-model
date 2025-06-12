# ai-model

# 📊 Customer Satisfaction Tracker

This project provides a smart tool to **track and evaluate customer satisfaction** during conversations using LLM-based analysis via [OpenRouter.ai](https://openrouter.ai). It consists of two main components:

1. `satisfaction_tracker.py` – A Python class for tracking satisfaction score based on dialogue.
2. `interactive_chat.py` – A simple Streamlit-based web app interface.

---

## 🚀 Features

* Tracks satisfaction on a scale of 1 (very dissatisfied) to 5 (very satisfied).
* Ignores greetings and uninformative inputs.
* Updates a running conversation summary.
* Provides automatic JSON feedback from an LLM.
* Offers a web interface for real-time interaction.

---

## 📂 File Structure

```
.
├── interactive_chat.py      # Streamlit frontend for chat and score display
├── satisfaction_tracker.py  # Backend logic for satisfaction tracking
└── README.md                # Project documentation
```
---

## 🛠️ Requirements

* Python 3.13.X
* `requests`
* `streamlit`

Install dependencies:

manually:

```bash
pip install requests streamlit
```

---

## 🔐 API Key

To use the app, you'll need an API key from [OpenRouter](https://openrouter.ai).

---

## ▶️ How to Run

1. Make sure both files are in the same directory.
2. Run the app with:

```bash
streamlit run app.py
```

3. Enter your OpenRouter API key.
4. Start chatting and tracking satisfaction!


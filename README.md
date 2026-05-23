# 🌍 Language Detection System using RNN

## 📌 Project Overview

This project is a deep learning-based **Language Detection System** built using a **Recurrent Neural Network (RNN)**. The model predicts the language of a given text sentence using Natural Language Processing (NLP) techniques.

The project demonstrates a complete machine learning workflow including:

- Data preprocessing
- Model training and evaluation
- Prediction pipeline
- Interactive Streamlit web interface

The application currently runs locally using Streamlit.

---

# 🧠 Model Building & Training

- Built using **TensorFlow (Keras)**
- Trained on a multilingual text dataset
- Implemented:
  - Text tokenization
  - Sequence padding
  - Label encoding
  - Class weight balancing for imbalanced data
- Used a **Simple RNN** architecture for sequential text learning
- Saved trained model in `.h5` format
- Serialized tokenizer and label encoder using `pickle`

---

# 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **95.96%** |
| Test Loss | **0.1505** |
| Class Weights | Enabled |

The model generalizes effectively on unseen multilingual text data.

---

# 🔍 Prediction Pipeline

The prediction system performs:

1. Text preprocessing  
2. Tokenization using saved tokenizer  
3. Sequence padding  
4. Language prediction using trained RNN model  
5. Confidence score generation  

---

# 🌐 Streamlit Application

A local Streamlit web application was developed for real-time predictions.

## Features

- Real-time language detection
- User text input
- Confidence score display
- Fast model loading using caching
- Simple and interactive UI

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP & ML | Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Streamlit |
| Model Persistence | Pickle |
| Development Tools | Jupyter Notebook |

---

# 📁 Project Structure

```bash
├── app.py
├── eda.ipynb
├── prediction.ipynb
├── requirements.txt
├── .gitignore
├── pic.png
├── saved_model/
│   ├── simple_rnn_model.h5
│   └── tokenizer.pkl
```

---

# ✅ Conclusion

This project demonstrates an end-to-end NLP and deep learning workflow for multilingual language detection using RNNs. The system achieves high accuracy and provides an interactive interface for real-time predictions.

---

# 🚀 Future Improvements

- Add support for more languages
- Replace Simple RNN with LSTM/GRU
- Display probability distribution for all languages
- Deploy the application online
- Improve UI/UX

---

# 👤 Author

**Arpan Mandal**  
B.Tech in Computer Science Engineering  
Haldia Institute of Technology  
West Bengal, India
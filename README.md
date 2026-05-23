# 🌍 Language Detection Using Recurrent Neural Network (RNN)

## 📌 Project Overview

Language detection is an important Natural Language Processing (NLP) task used in applications such as translation systems, search engines, and multilingual chatbots.  
This project implements a **Recurrent Neural Network (RNN)** model to automatically **detect the language of a given text sentence**.

The system is designed as an **end-to-end deep learning application**, starting from data analysis and model training to deployment using a web interface.

The system is divided into three main components:

1. **Model Building & Training**
2. **Prediction Pipeline**
3. **Streamlit Web Application**

---

## 🧠 1. Model Building & Training

- The model is built using **TensorFlow (Keras)**.
- A **multilingual text dataset** is used containing sentences from different languages.
- To handle **class imbalance**, the model is trained using **class weights**.
- Text preprocessing includes:
  - Tokenization of text data
  - Padding sequences to a fixed length
  - Label encoding of language classes
- A **Simple RNN** model is trained to learn sequential patterns in text.
- The trained model is saved in **`.h5`** format.
- The tokenizer and label encoder are serialized using **pickle** for reuse during prediction.

### 📊 Model Training & Evaluation Results

The model achieved strong performance on unseen test data:

- **Training with Class Weights:** Enabled  
- **Test Accuracy:** **95.96%**  
- **Test Loss:** **0.1505**

> These results indicate that the RNN model generalizes well and effectively captures language-specific patterns from text sequences.

### 📁 Relevant Files

- `eda.ipynb` – Exploratory Data Analysis on text data  
- `prediction.ipynb` – Model training and evaluation  
- `saved_model/simple_rnn_model.h5` – Trained RNN model  
- `saved_model/tokenizer.pkl` – Saved tokenizer & label encoder  

---

## 🔍 2. Prediction Module

- User input text is taken in real time.
- The input text is:
  - Converted into sequences using the saved tokenizer
  - Padded to the required input length
- The processed text is passed to the trained RNN model.
- The model outputs:
  - **Predicted language**
  - **Confidence score (probability)**

This ensures that the same preprocessing steps used during training are applied during inference.

---

## 🌐 3. Streamlit Web Application

An interactive **Streamlit-based web application** is developed to demonstrate the model.

### Features:
- Text input area for user sentences
- Example sentences for guidance
- Real-time language prediction
- Confidence score display
- Cached model loading for faster performance

The Streamlit application:
- Loads the trained RNN model
- Loads the saved tokenizer and label encoder
- Accepts user input and displays prediction results instantly

### ▶️ Run Locally

```bash
streamlit run app.py

| Category             | Technologies                |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| NLP & ML             | TensorFlow, Keras           |
| Data Processing      | NumPy, Pandas               |
| Machine Learning     | Scikit-learn                |
| Visualization        | Matplotlib, Seaborn         |
| Model Persistence    | Pickle                      |
| Web Framework        | Streamlit                   |
| Development Tools    | Jupyter Notebook, IPykernel |
```
📄 All dependencies are listed in requirements.txt

📁 Project Structure
```
├── app.py                         # Streamlit application
├── eda.ipynb                      # Exploratory Data Analysis
├── prediction.ipynb               # Model training & prediction
├── requirements.txt               # Project dependencies
├── .gitignore                     # Ignored files
├── pic.png                        # UI banner image
├── saved_model/
│   ├── simple_rnn_model.h5        # Trained RNN model
│   └── tokenizer.pkl              # Tokenizer & label encoder
```

## ✅ Conclusion

This project demonstrates a complete NLP-based deep learning system, covering text preprocessing, RNN model training, and deployment using a web application.

The use of class weights improves model robustness by handling class imbalance, resulting in high accuracy and reliable predictions.

## 🚀 Future Enhancements

-Add more languages to the dataset

-Replace RNN with LSTM / GRU for better accuracy

Display probability distribution for all languages

-Deploy the application on Streamlit Cloud

-Improve UI and user experience

👤 Author Details

Author: Arpan Mandal

Degree: B.Tech in Computer Science Engineering

Institute: Haldia Institute of Technology

Location: West Bengal, India

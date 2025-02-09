# 🌍 Arabic Dialect Identification | تصنيف اللهجات العربية 📝  
  
> **Project Type:** Natural Language Processing (NLP)  
> **Objective:** Classifying Arabic dialects using **Machine Learning (ML) & Deep Learning (DL)**  


---

## 🚀 Project Overview  

Arabic is a diverse language with many dialects that vary across regions. This project aims to develop a **Dialect Identification System** using **NLP techniques**, leveraging datasets from **QADI & MADAR**. The goal is to classify Arabic dialects efficiently using machine learning and deep learning models.

🔹 **Datasets Used:** QADI + MADAR (merged and preprocessed)  
🔹 **Models Implemented:** ML (SVM, Logistic Regression) & DL (LSTMs, Transformers)  
🔹 **Libraries Used:** `TensorFlow`, `PyTorch`, `Scikit-learn`, `NLTK`, `wordcloud`, `arabic-reshaper`  

---

## 🛠 Installation  

Before running the notebook, install the required dependencies:  

```bash
pip install datasets rarfile wordcloud arabic-reshaper python-bidi numpy pandas scikit-learn tensorflow torch nltk
```

## 📜 Dataset 

We use the QADI and MADAR datasets, which contain labeled Arabic dialectic texts spanning various regions.
Preprocessing includes:
✔ Tokenization using TweetTokenizer
✔ Stop-word removal & normalization
✔ Feature extraction (TF-IDF, word embeddings)

```bash
from datasets import load_dataset
dataset = load_dataset("arabic_dialect_dataset")
dataset = dataset.shuffle(seed=42)
```


## 🔬 Model Workflow 

1️⃣ Data Preprocessing
```bash
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

def preprocess(text):
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

dataset['train'] = dataset['train'].map(lambda x: {"text": preprocess(x["text"])})
```


## 2️⃣ Feature Engineering  

```bash
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(dataset['train']['text'])
```

## 3️⃣ Model Training
Logistic Regression Model

```bash
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train_tfidf, dataset['train']['label'])
```
Deep Learning (LSTM)

```bash
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense

model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(5, activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

## 4️⃣ Evaluation
```bash
y_pred = model.predict(X_train_tfidf)
accuracy = accuracy_score(dataset['train']['label'], y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```



🤝 Contributing
<br> Contributions are welcome! 


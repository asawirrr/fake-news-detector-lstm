# 📰 Fake News Detection System using LSTM & NLP

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK%20%7C%20Scikit--Learn-brightgreen.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning & Deep Learning system designed to identify and classify fake news articles with high confidence. The project combines Natural Language Processing (NLP) text preprocessing, a recurrent neural network architecture (**Long Short-Term Memory - LSTM**), and an interactive **Flask Web Interface** for real-time predictions.

---

## 📌 Project Overview

In an era of rapid digital information sharing, the spread of misinformation and fabricated news presents a major societal challenge. This project provides an automated verification tool that analyzes the linguistic patterns, semantic structures, and vocabulary of news articles to determine their authenticity (**Real News** vs. **Fake News**).

### ✨ Key Features
- **Comprehensive NLP Pipeline:** Stopwords removal, lemmatization with WordNet, punctuation and special character cleaning.
- **Deep Sequence Modeling (LSTM):** Captures long-range word dependencies and context far better than traditional bag-of-words methods.
- **Real-Time Web Application:** Built using Flask, allowing users to paste any article text and immediately view authenticity results along with a confidence probability.
- **Comparative Baseline Exploration:** Includes baseline benchmarks against classical machine learning algorithms (e.g., Naive Bayes and Random Forest).

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A[Raw News Article Text / Title] --> B[NLP Preprocessing: Lowercase, Regex, Stopwords, Lemmatization]
    B --> C[Tokenization & Sequence Padding (Max Length: 500)]
    C --> D[Embedding Layer (dim: 100, vocab: 50,000)]
    D --> E[SpatialDropout1D (0.2)]
    E --> F[LSTM Layer (100 units, recurrent dropout: 0.2)]
    F --> G[Dense Layer (Sigmoid Activation)]
    G --> H[Output Probability (0.0 to 1.0)]
    H --> I{Threshold Check (> 0.5)}
    I -->|True| J[Real News]
    I -->|False| K[Fake News]
```

---

## 🧠 Model Architecture

The core deep learning model is built using TensorFlow/Keras:

| Layer | Type | Configuration / Output Shape | Description |
| :--- | :--- | :--- | :--- |
| **1** | `Embedding` | `vocab_size=50000, dim=100` | Maps word tokens into dense 100-dimensional vector space |
| **2** | `SpatialDropout1D` | `rate=0.2` | Drops 1D feature maps to prevent co-adaptation of words |
| **3** | `LSTM` | `units=100, dropout=0.2, recurrent_dropout=0.2` | Recurrent layer learning temporal & sequential patterns |
| **4** | `Dense` | `units=1, activation='sigmoid'` | Binary output indicating probability of being Real News |

- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `adam`
- **Metric:** `accuracy`

---

## 📂 Repository Structure

```plaintext
├── Fake.csv                  # Dataset containing fake news articles
├── True.csv                  # Dataset containing authentic news articles
├── app_lstm.py               # Flask web application for inference & UI
├── train_lstm.py             # Model training & preprocessing pipeline
├── lstm_model.keras          # Serialized trained LSTM model (Git LFS)
├── tokenizer.pkl             # Serialized Keras Tokenizer (Git LFS)
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html            # Frontend web UI for Flask
├── .gitignore                # Ignored files (caches, checkpoints)
├── .gitattributes            # Git LFS configuration for binary files
└── README.md                 # Project documentation
```

---

## 📊 Dataset Details

The model is trained on the benchmark **ISOT Fake News Dataset** (included in this repository):
- **`Fake.csv`**: Contains thousands of fabricated articles labeled as `0`.
- **`True.csv`**: Contains verified, authentic news reports labeled as `1`.

Both datasets contain article attributes: `title`, `text`, `subject`, and `date`. The pipeline merges both sources, shuffles them uniformly, and performs binary classification.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/asawirrr/fake-news-detector-lstm.git
cd fake-news-detector-lstm
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
Run a quick Python command to download stopwords and wordnet:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🖥️ Running the Application

### Option A: Launch the Web App (Pre-trained Model)
To launch the Flask interface using the pre-trained weights:
```bash
python app_lstm.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Option B: Retrain the Model from Scratch
Run the training pipeline directly:
```bash
python train_lstm.py
```
This will retrain the LSTM network, evaluate classification metrics, and save new `lstm_model.keras` and `tokenizer.pkl` files.

---

## 📈 Evaluation & Results

The LSTM network achieves high accuracy and strong generalization across evaluation metrics:

- **Accuracy:** ~99%+ on test split
- **Evaluation Metrics:** Precision, Recall, F1-Score, and AUC-ROC score logged via Scikit-Learn `classification_report` and `roc_auc_score`.

---

## 🔮 Future Enhancements
- [ ] Add URL scraping to analyze online news links directly.
- [ ] Upgrade to Transformer-based models (e.g., BERT / RoBERTa) for comparative evaluation.
- [ ] Redesign the web frontend with modern CSS / glassmorphism and real-time visualization charts.
- [ ] Deploy to cloud services (e.g., Render, Hugging Face Spaces, or AWS).

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
Developed by **Asawir Hussain**  
- GitHub: [@asawirrr](https://github.com/asawirrr)

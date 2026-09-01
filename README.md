# Multilingual Emotion Detection

This project implements a multilingual multi-label emotion classifier using DistilBERT (English) and mBERT (multilingual) with LSTM+attention in PyTorch and Hugging Face Transformers. The model improves emotion detection performance across languages while handling class imbalance.

## Features
* **Multi-Language Support:** Processes and classifies text in English (`en`), Hindi (`hi`), and Marathi (`mr`).
* **Custom Architecture:** Enhances standard transformer outputs with a bidirectional LSTM and a custom attention layer.
* **Imbalance Handling:** Addresses dataset class imbalance using advanced loss functions, including Focal Loss, Asymmetric Focal Loss, and Weighted BCE Loss.
* **Model Interpretability:** Integrates LIME (Local Interpretable Model-agnostic Explanations) and attention weight heatmaps to explain model predictions word-by-word[cite: 1].
* **Automated Data Pipeline:** Automatically extracts, verifies, and formats the SemEval 2025 Task 11 dataset[cite: 1].

## Performance
* **English Model:** Improved F1-macro score from a baseline of 0.372 to **0.620**.
* **Multilingual Model:** Achieved an F1-macro score of **0.811**.

## Installation
```bash
pip install torch transformers scikit-learn numpy pandas matplotlib seaborn lime

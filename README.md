# NLP Text Classification: MLP vs. Transformer

This repository contains the implementation and analysis for a Natural Language Processing (NLP) project focused on text classification using the 20 Newsgroups dataset.

## Project Overview
The project compares three different architectures for classifying text into four categories:
1. **LogLinear Model (Single-Layer Perceptron)**: A baseline model using TF-IDF features.
2. **MLP Model (Multi-Layer Perceptron)**: A neural network with a hidden layer of 500 units.
3. **Transformer Model (DistilRoBERTa)**: A pre-trained transformer model fine-tuned for sequence classification.

## Key Findings
- The **Transformer model** achieved the highest accuracy (above 0.88).
- The **MLP model** showed significant sensitivity to the size of the training data portion.
- Detailed analysis of parameter counts and model performance is included in the provided PDF report.

## Files
- `ex2.py`: Main script for training and evaluating the models.
- `transformer_helpers.py`: Helper functions for the transformer implementation.
- `NaturalLanguageProcessing–Ex2.pdf`: Detailed project report and analysis.

## Dataset
The project uses the `20newsgroups` dataset from `sklearn.datasets`.

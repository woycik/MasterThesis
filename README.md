# MasterThesis

# Analysis of Selected Hallucination Detection Methods in LLMs 

Master's Thesis repository focused on detecting hallucinations in Large Language Models (LLMs) using **white-box methods** (probing internal representations, hidden states, and token distributions).

## Overview

LLMs are fluent, but they lie. This project tests the hypothesis that **hallucinations leave measurable footprints inside the model's brain** before the text is even fully generated. 

Instead of prompting another LLM to check the output (which is expensive and slow), this method extracts **8 interpretable internal features** directly from the transformer layers during inference.

### Extracted Features Include:
* **Representation Drift** (between layers and tokens)
* **Token Entropy & Model Uncertainty**
* **Context Centroid Distance** (how far a token drifts from the local context in the embedding space)
* **Prediction spread** of candidates in the unembedding space.

## Key Results & Tech Stack

* **Base Model:** `Llama-3.1-8B`
* **Datasets:** TriviaQA, SQuAD, NQOpen
* **Evaluation Metrics:** BLEURT, ROUGE-L (used for pseudo-labeling)
* **Top Performance:** Reached **0.897 AUROC** on TriviaQA.
* **Generalization:** Out-of-domain transfer scores stay strong (up to 0.746 AUROC), proving these internal signals are robust and competitive with state-of-the-art white-box detection methods.

## 🛠️ How it Works (Pipeline)

1. **Extraction:** Hook into Llama-3.1 layers during generation -> Extract an $N \times M$ feature matrix (where $N$ is response length, $M$ is features).
2. **Aggregation:** Compress the variable-length matrix into a fixed-size vector using `mean`, `std`, and `max` pooling.
3. **Classification:** Train a lightweight classifier to predict whether the model is hallucinating or telling the truth.

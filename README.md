# Toxic Content Classification using DistilBERT

This project focuses on classifying toxic content into categories using a multi-modal text input composed of user queries and associated image descriptions. The model is fine-tuned using a parameter-efficient approach (LoRA) on top of a pre-trained DistilBERT transformer.

## 📁 Dataset

The dataset (`toxic_data.csv`) contains the following columns:
- `query`: User-entered text.
- `image descriptions`: Textual descriptions associated with images.
- `Toxic Category`: The target label describing the type of toxicity.

## ⚙️ Preprocessing

- Cleaned text by removing punctuation, converting to lowercase, and trimming whitespace.
- Removed rows with missing or duplicate values.
- Encoded `Toxic Category` into numerical labels.
- Saved cleaned data as `cleaned.csv`.

## 🧠 Model

- Pretrained model: `distilbert-base-uncased`
- Tokenization of both `query` and `image descriptions`.
- Fine-tuning with **LoRA (Low-Rank Adaptation)** using the `peft` library.
- Built using `PyTorch` and `Transformers`.

## 🧪 Training and Evaluation

- Data split into training and validation sets (80/20).
- Wrapped tokenized inputs in a PyTorch `Dataset` class.
- Training performed with a custom training loop or `Trainer` (optional to be added).
- Evaluation metrics include accuracy, F1 score, and classification report.

## 🛠️ Dependencies

Install the required packages with:

```bash
pip install transformers datasets peft accelerate evaluate scikit-learn
```
## 📊 Results
After training, the model achieves high performance on toxic content classification, evaluated using F1-score and confusion matrices (visuals can be added if desired).

## 📂 Project Structure
```bash
├── cellula_task2.ipynb        # Main notebook
├── toxic_data.csv             # Raw data file
├── cleaned.csv                # Preprocessed data
├── README.md                  # Project description
```
## ✅ Future Improvements
Integrate image features to extend the multi-modal model.

Try other transformer models like BERT, RoBERTa.

Deploy model as an API for real-time toxicity detection.

## 🔒 License
This project is for educational and non-commercial use only.

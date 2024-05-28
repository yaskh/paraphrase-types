import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset

# Load models and tokenizers for BERT and RoBERTa
bert_model_path = '/Users/yasir/github/gipplab/paraphrase-types/out/cls-models/bert-large-uncased-jpwahle/etpc-paraphrase-detection/checkpoint-3045'
robert_model_path = '/Users/yasir/github/gipplab/paraphrase-types/out/cls-models/FacebookAI/roberta-base-jpwahle/etpc-paraphrase-detection/checkpoint-3045'

tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_path)
model_bert = AutoModel.from_pretrained(bert_model_path)

tokenizer_roberta = AutoTokenizer.from_pretrained(robert_model_path)
model_roberta = AutoModel.from_pretrained(robert_model_path)

def filter_data(train_df):
    # Convert each numpy array to tuple for hashability
    train_df['paraphrase_types'] = train_df['paraphrase_types'].apply(tuple)

    # train_df['paraphrase_types'].groupby(train_df['paraphrase_types']).count().index=
    # train_df['paraphrase_types'].groupby(train_df['paraphrase_types']).count().nlargest(3)

    reduced_to_similar_paraphrase_type = train_df[train_df['paraphrase_types'].apply(
                        lambda x: x == ("Same Polarity Substitution (contextual)", "Addition/Deletion", "Identity") )]
    return reduced_to_similar_paraphrase_type

def encode(sentences, model, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()

def compare_embeddings(dataset, model, tokenizer, n=15):
    top_n_indices = []

    for _, row in dataset.iterrows():
        embedding1 = encode(row['sentence1'], model, tokenizer)
        embedding2 = encode(row['sentence2'], model, tokenizer)

        difference = np.abs(embedding1 - embedding2)

        top_n_indices.extend(np.argsort(-difference)[:n])

    unique_values, counts = np.unique(top_n_indices, return_counts=True)
    top_10_indices = np.argsort(-counts)[:10]

    top_10_values = unique_values[top_10_indices]
    top_10_counts = counts[top_10_indices]

    return top_10_values, top_10_counts

def plot_top_10_bert_values(top_bert_10_values, top_10_bert_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(top_bert_10_values, top_10_bert_counts, color='blue')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.title('Top 10 indices with highest frequency of top n differences in BERT embeddings')
    plt.show()


df = load_dataset("jpwahle/etpc")['train'].to_pandas()
df = filter_data(df)
top_bert_10_values, top_10_bert_counts = compare_embeddings(df, model_bert, tokenizer_bert)
plot_top_10_bert_values(top_bert_10_values, top_10_bert_counts)
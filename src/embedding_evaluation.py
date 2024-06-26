import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset
from scipy.spatial import distance
import seaborn as sns

def load_model(model_path):
    tokenizer_bert = AutoTokenizer.from_pretrained(model_path)
    model_bert = AutoModel.from_pretrained(model_path)
    return model_bert, tokenizer_bert


def filter_data(train_df):
    # Convert each numpy array to tuple for hashability
    train_df['paraphrase_types'] = train_df['paraphrase_types'].apply(tuple)

    reduced_to_similar_paraphrase_type = train_df[train_df['paraphrase_types'].apply(
                        lambda x: x == ("Same Polarity Substitution (contextual)", "Addition/Deletion", "Identity") )]
    print(reduced_to_similar_paraphrase_type.shape)
    return reduced_to_similar_paraphrase_type

def encode(sentences, model, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_embeddings(dataset, model, tokenizer):
    embeddings_flattened = []
    batch_size = 32
    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]        
        batch_embedding1 = encode(batch['sentence1'], model, tokenizer)
        batch_embedding2 = encode(batch['sentence2'], model, tokenizer)
        for emb1, emb2 in zip(batch_embedding1, batch_embedding2):
            embeddings_flattened = [].append(torch.stack([emb1, emb2]))

    return np.array(embeddings_flattened)

def plot_top_10_values(top_bert_10_values, top_10_bert_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(top_bert_10_values, top_10_bert_counts, color='blue')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.title('Embedding component indices with highest frequency of differences in BERT embeddings for paraphrase type "Same Polarity Substitution (contextual)", "Addition/Deletion", "Identity"')
    plt.show()

def filter_for_single_pt(df):
    unique_paraphrase_types = set()
    df['paraphrase_types'].apply(lambda x: unique_paraphrase_types.update(x) if isinstance(x, (list, np.ndarray)) else unique_paraphrase_types.add(x))
    unique_paraphrase_types = list(unique_paraphrase_types)
    paraphrase_type_data_dict = {}
    
    for row in unique_paraphrase_types:
        reduced_to_similar_paraphrase_type = df[df['paraphrase_types']
                        .apply(lambda x: row in x if isinstance(x, (list, np.ndarray)) else row == x)]
        paraphrase_type_data_dict[row] = reduced_to_similar_paraphrase_type
        
    return paraphrase_type_data_dict


if __name__ == '__main__':
    df = load_dataset("jpwahle/etpc")['train'].to_pandas()
    single_type_dict = filter_for_single_pt(df)
    bert_model_path = '../out/cls-models/bert-large-uncased-jpwahle/etpc-paraphrase-detection/checkpoint-3045'
    bert_model, bert_tokenizer = load_model(bert_model_path)

    for key, value in single_type_dict.items():
        _, _, embeddings = calculate_embeddings(value, bert_model, bert_tokenizer)
        calculate_embeddings[key] = embeddings
        break
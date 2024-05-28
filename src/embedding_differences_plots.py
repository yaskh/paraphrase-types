import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset
from scipy.spatial import distance
import seaborn as sns
import time

mps_device = torch.device("mps")
mps_device = torch.device("cpu")
batch_size = 250

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
    start = time.time()
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input.to(mps_device))
    final_tensor = model_output.last_hidden_state.mean(dim=1).squeeze()
    print(f"Time taken to encode: {time.time() - start}")
    return final_tensor

def calculate_embeddings(dataset, model, tokenizer):
    embeddings_flattened = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]        
        batch_embedding1 = encode(batch['sentence1'].to_list(), model, tokenizer)
        batch_embedding2 = encode(batch['sentence2'].to_list(), model, tokenizer)
        for emb1, emb2 in zip(batch_embedding1, batch_embedding2):
            embeddings_flattened.append(torch.stack([emb1, emb2]))

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

def plot_hist(array_mean):
    window_size = 50
    averages = [np.mean(array_mean[i:i + window_size]) for i in range(0, len(array_mean), window_size)]
    indices = list(range(0, len(array_mean), window_size))

    plt.figure(figsize=(25, 10))
    plt.plot(indices, averages, color='blue', marker='o', linestyle='-')  # Connect points with a line
    plt.title('Scatter Plot of Average Values of Array Segments')
    plt.xlabel('Array Index')
    plt.ylabel('Average Value')
    plt.grid(True)
    plt.show()

files = glob.glob('src/out/embeddings/*.npy')
for file in files:
    data = np.load(file)
    print(data[1][1][1])
    result = data[:, 1, :] - data[:, 0, :]
    array_mean = np.mean(result, axis=0)
    plot_hist(array_mean)
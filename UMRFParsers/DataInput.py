import ExtractTrainingDescriptions
from Indexer import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import BertTokenizer, BertModel

examples = ExtractTrainingDescriptions.extract_training_exs()
labels = ExtractTrainingDescriptions.extract_training_labels()

# Construction of BERT embeddings on NL utterances
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
train_exs = [tokenizer(sent, return_tensors="pt") for sent in examples]
train_exs_embeds = [model(**x).last_hidden_state for x in train_exs]  # BERT Embeddings of NL input
# TODO: CONVERT train_exs_embeds from List[Tensor] to np array
# train_ex_embeds_np = torch.cat(train_exs_embeds)

# Different Embeddings for Output
labels_tokenized = [x.split() for x in labels]
output_indexer = Indexer()  # Contains the indexed numbers of the MR
for lab in labels_tokenized:
    for tok in lab:
        output_indexer.add_and_get_index(tok)
output_indexer.add_and_get_index("<UNK>")
labels_embed = []
for lab in labels_tokenized:
    labels_embed.append(index(lab, output_indexer))

# Line below fails because train_exs_embeds is not type nparray
# train_data = TensorDataset(torch.from_numpy(train_exs_embeds), torch.from_numpy(labels_embed))

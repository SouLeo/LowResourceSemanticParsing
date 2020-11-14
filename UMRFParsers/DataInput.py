import ExtractTrainingDescriptions
from Indexer import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import BertTokenizer, BertModel


examples = ExtractTrainingDescriptions.extract_training_exs()
labels = ExtractTrainingDescriptions.extract_training_labels()

# Construction of BERT embeddings on NL utterances (Input)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
train_exs = [tokenizer(sent, return_tensors="pt") for sent in examples]
train_exs_embeds = [torch.squeeze(model(**x).last_hidden_state) for x in train_exs]  # BERT Embeddings of NL input
padded_train_exs = pad_sequence(train_exs_embeds)
np_train = np.transpose(padded_train_exs.detach().numpy(), (1, 0, 2))

# Embeddings for Output (Output MRs)
labels_tokenized = [x.split() for x in labels]
output_indexer = Indexer()  # Contains the indexed numbers of the MR
for lab in labels_tokenized:
    for tok in lab:
        output_indexer.add_and_get_index(tok)
output_indexer.add_and_get_index("<UNK>")
labels_embed = []
for lab in labels_tokenized:
    labels_embed.append(index(lab, output_indexer))
list_len = [len(i) for i in labels_embed]
max_len = max(list_len)
np_labels = np.zeros((len(labels_embed), max_len))
for i in range(len(labels_embed)):
    t = max_len - len(labels_embed[i])
    np_labels[i, :] = np.pad(np.array(labels_embed[i]), pad_width=(0, t), mode='constant')

train_data = TensorDataset(torch.from_numpy(np_train), torch.from_numpy(np_labels), )
train_loader = DataLoader(train_data, shuffle=True, batch_size=4)

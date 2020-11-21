import ExtractTrainingDescriptions
from Indexer import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
from transformers import BertTokenizer, BertModel


def createTrainLoader():
    examples = ExtractTrainingDescriptions.extract_training_exs()
    labels = ExtractTrainingDescriptions.extract_training_labels()

    # Construction of BERT embeddings on NL utterances (Input)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    train_exs = tokenizer(examples, padding=True, return_tensors='pt')
    with torch.no_grad():
        train_exs_embeds = model(**train_exs).last_hidden_state  # BERT Embeddings of NL input

    input_ids = [101, 101, 101, 101]
    with torch.no_grad():
        sos_seq = model(torch.tensor(input_ids).unsqueeze(0)).last_hidden_state

    # Embeddings for Output (Output MRs)
    input_word_counts = Counter()

    labels_tokenized = [x.split() for x in labels]
    for sent in labels_tokenized:
        for word in sent:
            input_word_counts[word] += 1.0

    output_indexer = Indexer()  # Contains the indexed numbers of the MR
    for word in input_word_counts.keys():
        if input_word_counts[word] > 10:
            output_indexer.add_and_get_index(word)
    # for lab in labels_tokenized:
    #     for tok in lab:
    #         output_indexer.add_and_get_index(tok)
    output_indexer.add_and_get_index("<UNK>")
    labels_embed = []
    for lab in labels_tokenized:
        labels_embed.append(index(lab, output_indexer))
    list_len = [len(i) for i in labels_embed]
    max_len = max(list_len)
    np_labels = np.zeros((len(labels_embed), max_len))
    for i in range(len(labels_embed)):
        t = max_len - len(labels_embed[i])
        np_labels[i, :] = np.pad(np.array(labels_embed[i], dtype=np.int_), pad_width=(0, t), mode='constant')
    np_labels = np_labels.astype(int)

    train_data = TensorDataset(train_exs_embeds[:80], torch.from_numpy(np_labels[:80]))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=4)
    test_loader = DataLoader(train_data, shuffle=True, batch_size=len(train_exs_embeds))
    return train_loader, output_indexer, sos_seq, test_loader
    # return np_labels, train_exs_embeds

# Models TODO:
# 1) Seq2Seq with Bahandau Attention
# 2) Seq2Seq with Copy Mechanism
# 3) Pointer Network
# 4) Sequicity
# 5) Variational Autoencoder

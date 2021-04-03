import ExtractTrainingDescriptions
from Indexer import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
from transformers import BertTokenizer, BertModel
import spacy


def createTrainLoader():
    examples = ExtractTrainingDescriptions.extract_training_exs()
    labels = ExtractTrainingDescriptions.extract_training_labels()

    tsv_out = []
    for i in range(len(examples)):
        tsv_out.append(examples[i]+'\t'+labels[i])

    with open('UMRF_train.tsv', 'w') as file:
        for example in tsv_out:
            file.write('%s\n' % example)

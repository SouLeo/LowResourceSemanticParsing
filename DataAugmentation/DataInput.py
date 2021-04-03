import ExtractTrainingDescriptions
from Indexer import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
from transformers import BertTokenizer, BertModel
import spacy
import json


def createTrainLoader():
    examples = ExtractTrainingDescriptions.extract_training_exs()
    gold_examples_nodes = ExtractTrainingDescriptions.extract_training_nodes_exs()
    labels = ExtractTrainingDescriptions.extract_training_node_labels()


    # # Below is the automated script for generating frames. to use the manual generations, use the extract_training_nodes_exs function
    # nlp = spacy.load('en_core_web_sm')
    # node_list = []
    #
    # for ex in examples:
    #     doc = nlp(ex)
    #     annotation = []
    #     for token in doc:
    #         if token.text.lower() == 'left' or token.text.lower() == 'right' or token.text.lower() == 'look':
    #             annotation.append(0)
    #         elif token.text.endswith('ing') or token.text.endswith('ed'):
    #             if token.text.lower() == 'turning':
    #                 annotation.append(1)
    #             else:
    #                 annotation.append(0)
    #         elif token.pos_ == 'VERB' or token.text.lower() == 'walk' or token.text.lower() == 'turn':
    #             annotation.append(1)
    #         else:
    #             annotation.append(0)
    #     # len_of_doc = len(annotation) + 1
    #     annotation.append(1)
    #     verb_indices = np.asarray(np.nonzero(annotation)).squeeze()
    #
    #     pp = verb_indices.size
    #     for x in range(verb_indices.size-1):
    #         a = verb_indices[x]
    #         b = verb_indices[x+1]
    #         beans = doc[a:b]
    #         node_list.append(str(beans))
    #
    #
    # previous_indices = []
    # correct_examples = [None] * len(gold_examples_nodes)
    # for i in range(len(node_list)):
    #     for j in range(len(gold_examples_nodes)):
    #         if node_list[i] in gold_examples_nodes[j] or gold_examples_nodes[j] in node_list[i]:
    #             if j in previous_indices:
    #                 continue
    #             else:
    #                 correct_examples[j] = node_list[i]
    #                 previous_indices.append(j)
    #             break
    #     if j > len(gold_examples_nodes):
    #         correct_examples.append('ERR')
    #
    # errs = 0
    # for i in range(len(correct_examples)):
    #     if correct_examples[i] is None:
    #         correct_examples[i] = 'ERR'
    #         errs = errs + 1
    #
    # data_exs = correct_examples
    # data_labels = labels

    tsv_out = []
    # reverse
    for i in range(len(gold_examples_nodes[:1000])):
        if gold_examples_nodes[i] == 'ERR':
            continue
        tsv_out.append(labels[i]+'\t'+gold_examples_nodes[i])

    with open('UMRF_train_node.tsv', 'w') as file:
        for example in tsv_out:
            file.write('%s\n' % example)

    tsv_out_valid = []
    for z in range(len(gold_examples_nodes[1000:])):
        if gold_examples_nodes[1000+z] == 'ERR':
            continue
        tsv_out_valid.append(labels[1000+z]+'\t'+gold_examples_nodes[1000+z])

    with open('UMRF_valid_node.tsv', 'w') as file:
        for example in tsv_out_valid:
            file.write('%s\n' % example)

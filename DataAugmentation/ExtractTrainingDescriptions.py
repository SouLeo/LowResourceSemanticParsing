import pathlib
from pathlib import Path
import json

# Traverse all json data files in concat_graphs
# extract the graph description field, concat into training example list


def extract_training_nodes_exs():
    cwd = pathlib.Path().absolute()
    base_path = Path('concat_graphs')
    concat_graphs_dir = cwd / base_path
    filenames = list(concat_graphs_dir.glob('*.umrf.json'))

    training_sentences = []
    for file in filenames:
        with open(file) as infile:
            contents = json.load(infile)
            umrf_list = contents['umrf_actions']
            for umrf in umrf_list:
                training_sentences.append(umrf['description'])
    return training_sentences


def extract_training_node_labels():
    cwd = pathlib.Path().absolute()
    base_path = Path('concat_graphs')
    concat_graphs_dir = cwd / base_path
    filenames = list(concat_graphs_dir.glob('*.umrf.json'))

    training_labels = []
    for file in filenames:
        with open(file) as infile:
            contents = json.load(infile)
            umrf_list = contents['umrf_actions']
            for umrf in umrf_list:
                del umrf["name"]
                del umrf["id"]
                del umrf["effect"]
                del umrf["description"]

                try:
                    del umrf["children"]
                except:
                    print('no children')
                try:
                    del umrf["parents"]
                except:
                    print('no parents')
                training_labels.append(json.dumps(umrf).replace('"', ''))
    return training_labels


def extract_training_exs():
    cwd = pathlib.Path().absolute()
    base_path = Path('concat_graphs')
    concat_graphs_dir = cwd / base_path
    filenames = list(concat_graphs_dir.glob('*.umrf.json'))

    training_sentences = []
    for file in filenames:
        with open(file) as infile:
            contents = json.load(infile)
            umrf_list = contents['umrf_actions']
            temp_action_list = []
            for umrf in umrf_list:
                temp_action_list.append(umrf['description'])
            joined_actions = ' '.join(temp_action_list)
            training_ex = ' '.join(joined_actions.split())
            training_ex = training_ex
            training_sentences.append(training_ex)
    return training_sentences


def extract_training_labels():
    cwd = pathlib.Path().absolute()
    base_path = Path('concat_graphs')
    concat_graphs_dir = cwd / base_path
    filenames = list(concat_graphs_dir.glob('*.umrf.json'))

    training_labels = []
    for file in filenames:
        with open(file) as infile:
            training_labels.append(json.dumps(json.load(infile)).replace('"', ''))
    return training_labels


#################################### for mismatch ########################


def extract_training_node_labels_mismatch():
    cwd = pathlib.Path().absolute()
    base_path = Path('concat_graphs')
    concat_graphs_dir = cwd / base_path
    filenames = list(concat_graphs_dir.glob('*.umrf.json'))

    training_labels = []
    for file in filenames:
        with open(file) as infile:
            contents = json.load(infile)
            umrf_list = contents['umrf_actions']
            for umrf in umrf_list:
                del umrf["name"]
                del umrf["id"]
                del umrf["effect"]
                del umrf["description"]

                try:
                    del umrf["children"]
                except:
                    print('no children')
                try:
                    del umrf["parents"]
                except:
                    print('no parents')
                training_labels.append(umrf)
    return training_labels

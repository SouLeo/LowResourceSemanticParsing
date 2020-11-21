import pathlib
from pathlib import Path
import json

# Traverse all json data files in concat_graphs
# extract the graph description field, concat into training example list


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
            training_labels.append(json.dumps(json.load(infile)))
    return training_labels

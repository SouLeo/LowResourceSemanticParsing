from sklearn.metrics import cohen_kappa_score
from pathlib import WindowsPath


def open_annotations(wpath: WindowsPath):
    raise NotImplemented()
    # TODO:
    # Goal: using a WindowsPath that leads to the location of the Student Annotation folder, extract the file paths of the .json files


def create_groups(annotation_dict: dict):
    # TODO:
    # Goal: Summarize the code below. Hint, use for loops for the repeated code sections. Ex: 17-24, 26-33, 35-43 effectively are the same code.

    group_list = []
    group_1 = {}
    group_2 = {}
    group_3 = {}

    group_1_keys = [x for x in annotation_dict.keys() if x == 'Sid' or x == 'Bethany']
    for x in group_1_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                labels.append(infile.read())
        group_1[x] = labels
    group_list.append(group_1)

    group_2_keys = [x for x in annotation_dict.keys() if x == 'Minsoo' or x == 'Cynthia']
    for x in group_2_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                labels.append(infile.read())
        group_2[x] = labels
    group_list.append(group_2)

    group_3_keys = [x for x in annotation_dict.keys() if x == 'Ian' or x == 'Arrash']
    for x in group_3_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                labels.append(infile.read())
        group_3[x] = labels
    group_list.append(group_3)
    return group_list


def calculate_cohen_kappa(group_list: dict):
    raise NotImplemented()
    # TODO:
    # Goal: Calculate Cohen's Kappa by importing the function provided by SciKit learn (already imported on the file)


if __name__ == '__main__':
    wpath = WindowsPath('c:/Users/Selma/Desktop/Student Annotations/')  # change this with the location of your student annotations folder
    annotation_dict = open_annotations(wpath)
    group_list = create_groups(annotation_dict)
    print(calculate_cohen_kappa(group_list))

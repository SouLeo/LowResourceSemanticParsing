from sklearn.metrics import cohen_kappa_score
from pathlib import WindowsPath
import json
from difflib import SequenceMatcher
import statistics

def open_annotations(wpath: WindowsPath):
    student_folders = [x for x in wpath.iterdir() if x.is_dir()]  # This will create a list of the directories with the URA student names. This is called List Comprehension.
    student_folders[3] = [i for i in student_folders[3].iterdir() if i.match('50 Annotations - First')][0]  # Ian added more directories, have to get the correct folder and flatten the list.
    dict_list = [list(x.glob('*.json')) for x in student_folders if x.glob('*.json')]
    dict_keys = [x.name for x in student_folders]
    annotation_dict = {}
    i = 0
    for key in dict_keys:
        annotation_dict[key] = dict_list[i]
        i = i + 1
    annotation_dict['Ian'] = annotation_dict['50 Annotations - First']  # correcting Ian's name
    del annotation_dict['50 Annotations - First']  # removing Ian's old folder
    return annotation_dict


def create_groups(annotation_dict: dict):  # grouping the annotations with partners. Can lines 26-39 be done more succintly?? (yes it can :) )
    group_list = []
    group_1 = {}
    group_2 = {}
    group_3 = {}

    group_1_keys = [x for x in annotation_dict.keys() if x == 'Sid' or x == 'Bethany']
    for x in group_1_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                contents = json.load(infile)
                del contents['graph_name']
                del contents['graph_state']
                del contents['graph_description']
                labels.append(json.dumps(contents))
        group_1[x] = labels
    group_list.append(group_1)


    group_2_keys = [x for x in annotation_dict.keys() if x == 'Minsoo' or x == 'Cynthia']
    for x in group_2_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                contents = json.load(infile)
                del contents['graph_name']
                del contents['graph_state']
                labels.append(json.dumps(contents))
                labels.append(infile.read())
        group_2[x] = labels
    group_list.append(group_2)

    group_3_keys = [x for x in annotation_dict.keys() if x == 'Ian' or x == 'Arrash']
    for x in group_3_keys:
        labels = []
        for file in annotation_dict[x]:
            with open(file, 'r') as infile:
                contents = json.load(infile)
                del contents['graph_name']
                del contents['graph_state']
                labels.append(json.dumps(contents))
                labels.append(infile.read())
        group_3[x] = labels
    group_list.append(group_3)
    return group_list


def align_list(group_list):
    beth = group_list[0]['Bethany']
    sid = group_list[0]['Sid']
    b_json = []
    s_json = []
    for b in beth:
        temp = json.loads(b)
        for i in temp['umrf_actions']:
            del i['name']
            del i['id']
            del i['effect']
            b_json.append(i)
    for s in sid:
        temp = json.loads(s)
        for i in temp['umrf_actions']:
            del i['name']
            del i['id']
            del i['effect']
            s_json.append(i)
    ##
    cyn = group_list[1]['Cynthia']
    min = group_list[1]['Minsoo']
    c_json = []
    m_json = []
    for b in cyn:
        if b:
            temp = json.loads(b)
            for i in temp['umrf_actions']:
                del i['name']
                del i['id']
                del i['effect']
                c_json.append(i)
    for s in min:
        if s:
            temp = json.loads(s)
            for i in temp['umrf_actions']:
                del i['name']
                del i['id']
                del i['effect']
                m_json.append(i)
    ##
    arrash = group_list[2]['Arrash']
    ian = group_list[2]['Ian']
    a_json = []
    i_json = []
    for b in arrash:
        if b:
            temp = json.loads(b)
            for i in temp['umrf_actions']:
                del i['name']
                del i['id']
                del i['effect']
                a_json.append(i)
    for s in ian:
        if s:
            temp = json.loads(s)
            for i in temp['umrf_actions']:
                del i['name']
                del i['id']
                del i['effect']
                i_json.append(i)



    print('hi')


def calculate_cohen_kappa(group_list: dict):
    group_list[2]['Ian'].pop(0)
    group_list[2]['Ian'].pop(0)


    cohen_kappa = []

    for x in group_list:
        json_stuff1 = []
        json_stuff2 = []
        y = []
        for i in x.keys():
            y.append(x[i])
        for m in y[0]:
            if m:
                json_stuff1.append(m)
        for m in y[1]:
            if m:
                json_stuff2.append(m)

        if len(json_stuff1) > len(json_stuff2):
            json_stuff1 = json_stuff1[:len(json_stuff2)]
        else:
            json_stuff2 = json_stuff2[:len(json_stuff1)]

        match_score = []
        for i in range(0, len(json_stuff1)):
            match_score.append(SequenceMatcher(None, json_stuff1[i], json_stuff2[i]).ratio())

        cohen_kappa.append(statistics.median(match_score))  # tell me why this raises an error and how to fix it
    return cohen_kappa


if __name__ == '__main__':
    wpath = WindowsPath('c:/Users/Selma/Desktop/Student Annotations/')   # change this with the location of your student annotations folder
    annotation_dict = open_annotations(wpath)
    group_list = create_groups(annotation_dict)
    aligned_list = align_list(group_list)
    print(calculate_cohen_kappa(group_list))

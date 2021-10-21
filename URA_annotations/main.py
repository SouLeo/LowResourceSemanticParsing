import pathlib
from pathlib import Path
import json

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cwd = pathlib.Path().absolute()
    base_umrf_path = Path('train')
    umrf_graphs_dir = cwd / base_umrf_path
    umrf_graph_list = [x for x in umrf_graphs_dir.iterdir() if x.is_dir()]
    # umrf_pick_and_place = [x for x in umrf_graph_list if x.match("pick_and_place_simple*")]
    umrf_look_at_obj = [x for x in umrf_graph_list if x.match("look_at_obj_in*")]

    trial_list = []
    for x in umrf_look_at_obj:
        for i in x.iterdir():
            if i.is_dir():
                trial_list.append(i)

    filenames = []
    for x in trial_list:
        p = Path(x).glob('**/*')
        for i in p:
            if i.is_file():
                filenames.append(i)

    annotations_list = []
    for file in filenames:
        with open(file) as infile:
            contents = json.load(infile)['turk_annotations']
            k = 0
            for i in contents:
                hd = contents['anns'][k]['high_descs']
                for j in hd:
                    annotations_list.append(j)
                td = contents['anns'][k]['task_desc']
                annotations_list.append(td)
                k = 1 + k


    # lists = [[] for _ in range(3)]
    # i = 0
    # for item in annotations_list:
    #     if i % 3 == 0:
    #         lists[0].append(item)
    #     elif i % 3 == 1:
    #         lists[1].append(item)
    #     else:
    #         lists[2].append(item)
    #     i = i + 1
    #
    # with open('group_1_annotation_list.txt', 'w') as f:
    #     for item in lists[0]:
    #         f.write("%s\n" % item)
    #
    # with open('group_2_annotation_list.txt', 'w') as f:
    #     for item in lists[1]:
    #         f.write("%s\n" % item)
    #
    # with open('group_3_annotation_list.txt', 'w') as f:
    #     for item in lists[2]:
    #         f.write("%s\n" % item)

    lists = [[] for _ in range(2)]
    i = 0
    for item in annotations_list:
        if i % 2 == 0:
            lists[0].append(item)
        elif i % 2 == 1:
            lists[1].append(item)
        i = i + 1

    with open('group_1_annotation_list_fall.txt', 'w') as f:
        for item in lists[0]:
            f.write("%s\n" % item)

    with open('group_2_annotation_list_fall.txt', 'w') as f:
        for item in lists[1]:
            f.write("%s\n" % item)

    # with open('group_3_annotation_list.txt', 'w') as f:
    #     for item in lists[2]:
    #         f.write("%s\n" % item)



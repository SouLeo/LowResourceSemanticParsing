import os
import pathlib
from pathlib import Path
import json

# This script will generate a concat_graphs folder in the same working directory as the script.

# traverse all the umrf_x directories in umrf_graphs, save the index 'x' for future file output
# concatenate all files within the umrf_x directory, then output to a separate output directory

cwd = pathlib.Path().absolute()
base_umrf_path = Path('umrf_graphs')
umrf_graphs_dir = cwd / base_umrf_path
base_umrf_concatgraphs = Path('concat_graphs')
umrf_output = cwd / base_umrf_concatgraphs

os.mkdir(umrf_output)

umrf_graph_list = [x for x in umrf_graphs_dir.iterdir() if x.is_dir()]

print(umrf_graph_list)

for x in umrf_graph_list:
    umrf_graph_name = os.path.basename(x)
    index = str(umrf_graph_name)[5:]  # Hardcoded and assumes an naming scheme of umrf_X
    graph_name = "fsm_graph_" + index
    filenames = list(x.glob('*.umrf.json'))
    new_graph_file = str(umrf_output / umrf_graph_name) + '.umrf.json'
    umrfgraph_jsons = {"graph_name": graph_name, "graph_description":""}  # DICTIONARY OF NEW JSON UMRF GRAPH TEMPLATE
    with open(new_graph_file, "w") as outfile:
        umrfnodes_jsons = []  # DICTIONARY OF JSON OBJECTS OF EXISTING UMRF NODES IN GRAPH
        for filename in filenames:
            with open(filename) as infile:
                contents = umrfnodes_jsons.append(json.load(infile))
        umrfgraph_jsons['umrf_actions'] = umrfnodes_jsons
        contents = json.dumps(umrfgraph_jsons)
        outfile.write(contents)

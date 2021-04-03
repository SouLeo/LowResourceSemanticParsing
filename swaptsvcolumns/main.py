# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fileinpath = 'C:/Users/Selma/PycharmProjects/swaptsvcolumns/augment_ablations_copynet/RI_RS_RD_UMRF_train_node.tsv'
    fileoutpath = 'C:/Users/Selma/PycharmProjects/swaptsvcolumns/fixed_copynet_ablations/14.tsv'
    with open(fileinpath, 'r') as fin, open(fileoutpath, 'w', newline='') as fout:
        freader = csv.reader(fin, delimiter='\t')
        fwriter = csv.writer(fout, delimiter='\t')
        for line in freader:
            line[1], line[0] = line[0], line[1]  # switches position between first and last column
            fwriter.writerow(line)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

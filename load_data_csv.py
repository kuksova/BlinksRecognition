import numpy as np

import csv

def load_data_csv(fname):
    csv_file = open(fname,'r')
    dataA = []
    dataB = []
    dataC = []

    # Read off and discard first line, to skip headers
    csv_file.readline()

    # Split columns while reading
    for a, b, c in csv.reader(csv_file, delimiter=','):
        # Append each variable to a separate list
        dataA.append(int(a))
        dataB.append(int(b))
        dataC.append(float(c))
    return dataC


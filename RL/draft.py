

import os
import pickle
import numpy as np
from utils import *


filename = "q_table.pickle"

def load_pickle():
    with open(filename, 'rb') as f:
        table = pickle.load(f)
    return table

def save_pickle(table):
    with open(filename, 'wb') as f:
        pickle.dump(table, f)


table = {"player": 34, "enemy": 4344}

newtable = load_pickle()
values = np.array(list(newtable.values())).reshape(-1)

fig = figure(figsize=(13, 7))
plt.hist(values, bins = len(np.unique(values))//1000)
fig.savefig("hist3.png")
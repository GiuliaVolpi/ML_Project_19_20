import sys
from os import path

sys.path.insert(0, "../")
sys.path.insert(0, "./")

from isanet.utils.model_utils import save_data
import numpy as np
import itertools

def k_fold(dataset, k, shuffle = True):
    
    index = []

    elem = np.arange(0, dataset.shape[0])
    
    if shuffle:
        np.random.shuffle(elem)

    num_elem = int(np.floor(dataset.shape[0]/k))

    for fold in range(0, k-1):
        start = fold*num_elem
        end = num_elem*(fold+1)
        index.append(np.array(elem[start:end]))
    start = (k-1)*num_elem
    index.append(np.array(elem[start:]))

    val_index = []
    train_index = []
    for val in range(0, k):
        list_temp = []
        for i in range(0, k):
            if val == i:
                val_index.append(index[i].tolist())
            else:
                list_temp.append(index[i].tolist())
        train_index.append(list(itertools.chain.from_iterable(list_temp)))
    
    
    return {"train": train_index, "val": val_index}

TS = np.genfromtxt('ML-CUP19-TR.csv',delimiter=',')

np.random.shuffle(TS)

np.savetxt('cup_1412/ML-CUP19-TR_test.csv', TS[1412:,1:], delimiter=',')
np.savetxt('cup_1412/ML-CUP19-TR_tr_vl.csv', TS[:1412,1:], delimiter=',')

split = k_fold(TS[1412:,1:], 4)
save_data(split, "cup_1412/4folds.index")
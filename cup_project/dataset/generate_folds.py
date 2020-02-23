import sys
from os import path

sys.path.insert(0, "../../")
sys.path.insert(0, "./")


from isanet.model_selection import Kfold
from isanet.utils.model_utils import save_data
import numpy as np

np.random.seed(42)
TS = np.genfromtxt('ML-CUP19-TR.csv',delimiter=',')

np.random.shuffle(TS)

np.savetxt('cup20/ML-CUP19-TR_tr_vl.csv', TS[:1500,1:], delimiter=',')
np.savetxt('cup20/ML-CUP19-TR_test.csv', TS[1500:,1:], delimiter=',')

kf = Kfold(n_splits=4, shuffle=True)
split = kf.split(TS[:1500,1:])
save_data(split, "cup20/4folds.index")

ten_features = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 21, 22]
np.savetxt('cup10/ML-CUP19-TR_tr_vl_10.csv', TS[:1500,ten_features], delimiter=',')
np.savetxt('cup10/ML-CUP19-TR_test_10.csv', TS[1500:,ten_features], delimiter=',')

save_data(split, "cup10/4folds.index")

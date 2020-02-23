import sys
from os import path

sys.path.insert(0, "../")
sys.path.insert(0, "./")


from isanet.model_selection import Kfold
from isanet.utils.model_utils import save_data
import numpy as np

TS = np.genfromtxt('ML-CUP19-TR.csv',delimiter=',')

np.random.shuffle(TS)

np.savetxt('cup20_tr_vl_ts_folds_fix/ML-CUP19-TR_test.csv', TS[1500:,1:], delimiter=',')
np.savetxt('cup20_tr_vl_ts_folds_fix/ML-CUP19-TR_tr_vl.csv', TS[:1500,1:], delimiter=',')

kf = Kfold(n_splits=5, shuffle=True, random_state=42)
split = kf.split(TS[:1500,1:])
save_data(split, "cup20_tr_vl_ts_folds/5folds.index")

ten_features = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 21, 22]
np.savetxt('cup10_tr_vl_ts_folds/ML-CUP19-TR_test_10.csv', TS[1500:,ten_features], delimiter=',')
np.savetxt('cup10_tr_vl_ts_folds/ML-CUP19-TR_tr_vl_10.csv', TS[:1500,ten_features], delimiter=',')

save_data(split, "cup10_tr_vl_ts_folds/5folds.index")

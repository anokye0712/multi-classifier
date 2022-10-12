from tqdm import tqdm
from scipy.special import softmax
import numpy as np
import skfuzzy as sk

######################################################################
######################################################################

def k(x):
    i = sk.sigmf(x, 0, -2)
    y = sk.sigmf(x, 0, 2)
    return np.maximum(i, y) #i + y - (i * y)

def ranks_np (x):    
    i = np.apply_along_axis(k, axis=1, arr=x)    
    return i #fuzzy ranks

def ensemble_np(probs, acc, alpha=2): 
    fuse_score = np.empty(probs[0].shape) #empty tensor of size row x column as prediction size
    total_acc = sum(acc) #sum of all accuracies

    rank = [ranks_np(prob) for prob in probs]
    for i in tqdm(range(fuse_score.shape[0])):
        vals = [] #empty set to hold a list of 1d tensor values
        assert len(probs) == len(acc), f'Length(model predictions): {len(probs)} is not equal to Length(accuracies): {len(acc)}'
        for value in zip(rank, acc):
            cal = np.power((value[1] / total_acc) * (value[0][i] - 0.5), alpha) #calc values for rank rows using the accuracy
            vals.append(cal)
        sum_ = sum(vals) #sum all values for 
        pr = 0.5 + np.power(sum_, 1 / alpha) #
        fuse_score[i: ] = pr
    return softmax(fuse_score, axis=1)

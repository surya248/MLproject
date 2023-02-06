import pandas as pd
import numpy as np
import tqdm

import utils as ut

def load_3D_from_csv(base_path='/home/mohit/ml_project/data',dtype='train'):
    '''
    Helper function to flatten out the data and convert each sample into 3D format
    
    Format of a sample: [user_id,question_id,is_correct]
    '''
    if dtype=='train':
        data_csv = ut.load_train_csv(base_path)
    elif dtype=='val':
        data_csv = ut.load_valid_csv(base_path)
    else:
        data_csv = ut.load_public_test_csv(base_path)

    n = len(data_csv['user_id'])
    data = []

    for i in range(n):
        data.append([data_csv['user_id'][i],
                    data_csv['question_id'][i],
                    data_csv['is_correct'][i]])
    data = np.array(data)

    return data

def evaluate_3D(pred,label):
    '''
    pred: [num_samples,is_correct]
    '''
    return float(sum(pred==label))/float(len(label))
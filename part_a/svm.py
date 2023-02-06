import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import utils as ut
import utils_partb as utb

base_path = '/home/mohit/ml_project/data'

train_data = utb.load_3D_from_csv(base_path)
val_data = utb.load_3D_from_csv(base_path,dtype='val')
test_data = utb.load_3D_from_csv(base_path,dtype='test')

clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto',cache_size=1000,verbose=True))

clf.fit(train_data[:,:2],train_data[:,2])

val_pred = clf.predict(val_data[:,:2])
val_acc = utb.evaluate_3D(val_pred,val_data[:,2])

test_pred = clf.predict(test_data[:,:2])
test_acc = utb.evaluate_3D(test_pred,test_data[:,2])

print(f'Final Validation Accuracy: {val_acc}')
print(f'Final Test Accuracy: {test_acc}')
from utils import *
import utils_partb as utb
import lightgbm as ltb
from sklearn import metrics

def train_onehot(base_path="../data"):
    train_data = utb.load_3D_from_csv(base_path,dtype='train')
    val_data = utb.load_3D_from_csv(base_path,dtype='val')
    test_data = utb.load_3D_from_csv(base_path,dtype='test')
    X_train = train_data[:,:2]
    y_train =train_data[:,2]
    X_test = test_data[:,:2]
    y_test =test_data[:,2]

    model = ltb.LGBMClassifier()
    model.fit(X_train, y_train)
    predicted_y = model.predict(X_train)
    print(metrics.classification_report(y_train, predicted_y))
    predicted_y = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_y))
    
if __name__ == "__main__":
    train_onehot()

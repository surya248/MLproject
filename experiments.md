# Notes
- Nueral Network is generalizing on basis of questions, not on the basis of students.
- SVM performs poorly because there is no no relation between corresponding a certain user and its next user, same can be said for item
- IMplemented Nueral Network with one hot encoding 
    - Better performance than standard nueral network
    - Overfitting slightly
    - Question Metadata and Student Metadata can also be passed through nueral network
    - Can be thought of as using information theory byt not strictly getting restricted to the function used

# SVM
- Using Default Config
- Results
    - Final Validation Accuracy: 0.6007620660457239
    - Final Test Accuracy: 0.5958227490826983

# KNN 
- With Default parameters mentioned in question
- Results
    - Best Test Accuracy based on colloborative user filtering : 68.417
    - Best Test Accuracy based on colloborative item filtering: 66.892


## Experiment - 1
- Changed weights from 'uniform' to 'distance' in KNNImputer, for both user based and item based filtering.
- Results
    - Best Test Accuracy based on colloborative user filtering : 67.090
    - Best Test Accuracy based on colloborative user filtering : 67.090

## Experiment -2
- Added Missing Tag indicator in KNN Imputer
- Results
    - Best Test Accuracy based on colloborative user filtering : 68.417
    - Best Test Accuracy based on colloborative item filtering: 66.892

# IRT 
Final Validation accuracy is 0.7081569291560824\
Final Test Accuracy is 0.7044877222692634

## Experiment - 1
- Increasing lr to 0.01
- Results
    - Final Validation accuracy is 0.7050522156364663
    - Final Test Accuracy is 0.7061812023708721

## Experiment - 2
- Increasing lr to 0.1
- Results
    - Final Validation accuracy is 0.6529777025119955
    - Final Test Accuracy is 0.6404177250917301

## Experiment - 3
- Increasing Iterations to 100
- Results
    - Final Validation accuracy is 0.7061812023708721
    - Final Test Accuracy is 0.7081569291560824

## Experiment - 4
- Increasing Iterations to 100 and lr to 0.01
- Results
    - Final Validation accuracy is 0.7061812023708721
    - Final Test Accuracy is 0.7090036692068868

## Experiment - 5
- init theta and beta from np.zeros instead of np.random.randn
- **Observed: Very early convergence**
- Results
    - Final Validation accuracy is 0.7061812023708721
    - Final Test Accuracy is 0.7050522156364663

## Experiment - 6
- init theta and beta from np.ones instead of np.random.randn
- **Observed:  Early convergence**
- Results
    - Final Validation accuracy is 0.7061812023708721
    - Final Test Accuracy is 0.7050522156364663


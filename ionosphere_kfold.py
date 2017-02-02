"""
Development and evaluation of deep learning model using k-fold cross validation.
this is a bbinary classification talk for innosphere dataset retrieved
 from https://archive.ics.uci.edu/ml/datasets/Ionosphere?__s=cmaatmzksgyarkpg3iqj
the method used here is to evaluating model performance and optimizing model hyperparameters using scikit-learn.
import libraries, and Keras Deep learning Framework
"""
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas  as pd


"""function to create, and innitialize model"""
def create_model():
    model = Sequential()  # create model
    # add fully connected layers around the area of the dataset
    model.add(Dense(350, input_dim=34, init='uniform', activation='relu'))
    model.add(Dense(35, input_dim=2, init='uniform', activation='relu'))  # specfy data attribute
    model.add(Dense(1, init='uniform', activation='sigmoid'))  # final layer
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
    # compile and run the model,  Learning schedules, adam, sgd
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

"""this defines the main function """
def main():
    # preprocess and read in data
    seed = 7
    np.random.seed(seed)
    df = pd.read_csv("dataset.csv", sep=",")
    dataset = np.array(df)

    # innitialize the X and Y values
    X = dataset[:, 0:34]
    Y = dataset[:, 34]

    # normalize the data, 0 for b and 1 for g
    Y = [0 if x is 'b' else 1 for x in Y]

    # reconvert the data from list to array
    Y = np.asarray(Y)

    # create the model
    model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=35, verbose=1)

    # evaluate the model using 10-fold (k-fold evaluation strategy)
    kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean()) #find the mean of the array of the result


"""redefine the main function"""
if __name__ == '__main__':
    main()

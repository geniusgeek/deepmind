"""
this is a bbinary classification talk for innosphere dataset retrieved
from https://archive.ics.uci.edu/ml/datasets/Ionosphere?__s=cmaatmzksgyarkpg3iqj
"""
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas  as pd


"""function to create, and innitialize model"""
def create_model():
    model = Sequential()  # create model
    # add fully connected layers around the area of the dataset
    model.add(Dense(350, input_dim=34, init='uniform', activation='relu'))
    model.add(Dense(35, input_dim=2, init='uniform', activation='relu'))  # specfy data attribute
    model.add(Dense(1, init='uniform', activation='sigmoid'))  # final layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile and run the model
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
    model = create_model()
    # add a model checkpoint to save the best model, mostly suitable for long running task
    checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_acc', save_weights_only=True, mode='auto',
                                 verbose=True)

    # fit the data and register a history callback
    history = model.fit(X, Y, nb_epoch=150, batch_size=35, callbacks=[checkpoint])
    # investigate the history object, examine the model performance and print the list of history metrics
    print(history.history.keys())

    scores = model.evaluate(X, Y)  # evaluate results and scores
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))  # print the percentage value of the score


"""redefine the main function"""
if __name__ == '__main__':
    main()

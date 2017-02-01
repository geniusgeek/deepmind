from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas  as pd

#preprocess and read in data
seed=7
np.random.seed(seed)
df=pd.read_csv("dataset.csv",sep=",")
dataset=np.array(df)

#innitialize the X and Y values
X=dataset[:,0:34]
Y=dataset[:,34]

model=Sequential() #create model
#add fully connected layers around the area of the dataset
model.add(Dense(350, input_dim=34,init='uniform',activation='relu'))
model.add(Dense(35, input_dim=2,init='uniform',activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid')) #final layer and output
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#compile and run the model
model.fit(X,Y,nb_epoch=150,batch_size=35) #fit the data

scores = model.evaluate(X, Y) #evaluate results and scores
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))#print the percentage value of the score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas  as pd

seed=7
np.random.seed(seed)
df=pd.read_csv("dataset.csv",sep=",")
dataset=np.array(df)

X=dataset[:,0:34]
Y=dataset[:,34]

model=Sequential()
model.add(Dense(350, input_dim=34,init='uniform',activation='relu'))
model.add(Dense(35, input_dim=2,init='uniform',activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,nb_epoch=150,batch_size=35)


scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
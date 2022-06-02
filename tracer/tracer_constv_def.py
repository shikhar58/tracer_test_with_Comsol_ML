import mph
import numpy as np

client = mph.start(cores=1)
model = client.load('test_tracer_API3.mph')

def comsolmodel(theta, disp):
    print("comsol model is running...")
    model.parameter('theta', theta)
    model.parameter('disp', disp)
    model.solve()
    #print(model.evaluate("c"))
    #MeshSequence = model.java.mesh("mesh1")
    #Coords = MeshSequence.getVertex()
    #NumericalFeatureList = model.java.result().numerical()
    #Interpolation = NumericalFeatureList.create("Inter", "Interp")
    #Interpolation.setInterpolationCoordinates(Coords)
    #Interpolation.set("expr","c")
    #conc = Interpolation.getData()
    c = model.evaluate(['c'])
    return(c)



n=10000
data=np.zeros((n,51))
theta_y=np.zeros(n)
disp_y=np.zeros(n)

for i in range(n):
    theta_y[i]=np.random.uniform(low=0.2, high=0.8)
    disp_y[i]=np.random.uniform(low=0.005, high=0.2)
    c = comsolmodel(theta_y[i], disp_y[i])
    c = np.array(c)
    c = np.reshape(c, (51, 134))
    c_btc = c[:, -1]
    data[i,:]=c_btc
    print("working {}".format(i))

theta_y=np.reshape(theta_y,(n,1))
disp_y=np.reshape(disp_y,(n,1))
dataset=np.concatenate((data,theta_y,disp_y), axis=1)
import pandas as pd
df = pd.DataFrame(dataset)
df.to_csv("final_data2.csv")


#ML MODEL

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("final_constv_def.csv", sep = ",")

#dataset=dataset.replace(0,np.nan).dropna(axis=1,how="all")

#dataset=dataset.replace(1,np.nan).dropna(axis=1,how="all")

dataset=dataset.iloc[:,1:]


X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(Dense(20, input_dim=51, activation='relu'))
#model.add(keras.layers.Dropout(0.5)) results get worse
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model, mean squared error works best
model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10)

y_pred = model.predict(X_test)

idx = 500
aa=[x for x in range(idx)]
plt.figure(figsize=(8,4))
plt.plot(aa, y_test[:idx], marker='.', label="actual")
plt.plot(aa, y_pred[:idx], 'r', label="prediction")

from sklearn.metrics import r2_score
a= r2_score(y_pred, y_test)
plt.scatter(y_pred, y_test)


#actual test

dataset_test = pd.read_csv("test_0.25_0.005_tracer.csv", sep = ",")

X_testone_v = dataset_test.iloc[:, -1].values

X_testone= np.reshape((X_testone), (1,51) )
y_predone = model.predict(X_testone)
plt.scatter(range(51), X_testone_v)
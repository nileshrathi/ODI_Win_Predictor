import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random

def mmn(data, min, max):
    return (data-min)/(max-min)


#lading data from npy file
#0th index: wickets in hand
#1st index: runs remaining/over remaining
#2nd index: label
data = np.load("11_22_onehot_training_data.npy", allow_pickle=True)
# d_min = np.amin(data, axis = 0)
# d_max = np.amax(data, axis = 0)
data1 = [np.array(d) for d in data]

data1 = np.array(data1)

random.shuffle(data1)
random.shuffle(data1)
random.shuffle(data1)

print(data1[0])
print(len(data1[0]))
X = []
y = []
X=data1[:,0:53].copy()
y=data1[:,[53]].copy()
#for d in data:
    #X.append([mmn(d[0], d_min[0], d_max[0]), mmn(d[1], d_min[1], d_max[1]), mmn(d[2], d_min[2], d_max[2]), mmn(d[3], d_min[3], d_max[3])])
    # X.append([mmn(d[0], d_min[0], d_max[0]), mmn(d[1], d_min[1], d_max[1]), mmn(d[2], d_min[2], d_max[2]), d[3]])
    
#    y.append([d.pop()])
    # d = [[d[i]] for i in range(len(d))]
#    X.append(d)


    

#divide the data into train and test
train_size = int(0.80*len(X))
test_size = len(X) - train_size

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[-test_size:]
y_test = y[-test_size:]




X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# l1s = [64, 128, 256, 512, 1024]
# l2s = [64, 128, 256, 512, 1024]
# batch_size = [256, 512, 1024, len(X_train)]

#print(data[0])

l1s = [512]
l2s = [256]
batch_size = [64]

# l1s = [64, 128, 256, 512, 1024, 2048]
# l2s = [64, 128, 256, 512, 1024, 2048]
# batch_size = [256, 512, 1024]

accuracy_list = []
hp = []

for l1 in l1s :
    for l2 in l2s:
        for bs in batch_size:

            hp.append(str(l1) + "_" + str(l2) + "_" + str(bs))

            # define the keras model
            model = Sequential()
            model.add(Dense(l1, input_dim=53, activation='relu'))
            model.add(Dense(l2, activation='relu'))
            model.add(Dense(1, activation='relu'))

            adam = keras.optimizers.Adam(learning_rate=0.001)

            # compile the keras model
            model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

            # fit the keras model on the dataset
            #model.fit(X_train, y_train, epochs=400, batch_size=len(X_train), validation_split=0.1)
            model.fit(X_train, y_train, epochs=100, batch_size=bs)

            # evaluate the keras model
            _, accuracy = model.evaluate(X_test, y_test)
            print('Accuracy: %.2f' % (accuracy*100))

            accuracy_list.append(accuracy*100)

            model.save("./models/model_w2_rd_" + str(l1) + "_" + str(l2) + "_" + str(bs) + ".nn")
            results = model.predict(X_test)
            # for i in range(len(X_test)):
            #     print(str(X_test[i]) + " : " + str(y_test[i]) + " : " + str(results[i]))

# print(max(accuracy_list))
# print(hp.index(accuracy_list.index(max(accuracy_list))))

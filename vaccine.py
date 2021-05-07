# %% Imports
import pickle as pk
import math
import csv
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% Loading data
test_data = pk.load(open("test.pickle", "rb"))
(train_data_1, train_labels_1) = pk.load(open("train1.pickle", "rb"))
(train_data_2, train_labels_2) = pk.load(open("train2.pickle", "rb"))

# %% Merge train data
total_train = train_data_1
total_train = total_train.append(train_data_2)

trainY = np.concatenate((train_labels_1, train_labels_2))

# %% Sub-train
train = total_train[:]

# Convert last 6 columns
train.iloc[:, -6:] = train.iloc[:, -6:]\
    .apply(lambda col:
           col
           .apply(lambda v: str(v)[2:])
           .apply(lambda x: -1.0 if x == 'n' else float(x))
           )


# %% Filter out null variances
variances = train.var()
train_var = train.iloc[:, np.array(variances) > 0]

# %% Mean 
means = train_var.mean()
train_var = train_var.fillna(means)
# %% Save train
# filename = "train_var_nan.pickle"
# with open(filename, "wb") as output_file:
#   pk.dump(train_var, output_file)
# %% Load the first prefiltered train data
# train_var = pk.load(open(filename, "rb"))

# %% Normalize
scaler = StandardScaler()
train_norm = pd.DataFrame(scaler.fit_transform(train_var))
train_norm.columns = train_var.columns

# %%Do every thing many times
big_scores_test = []
for big_it in range(1):
  print(big_it)
  # %% Randomize and clean data set
  trainXY = train_norm 
  trainXY['label'] = pd.Series(trainY, index=trainXY.index)
  trainXY = trainXY.sample(frac=1,random_state=big_it)

  lil_test_number = 50
  
  lil_test = trainXY.iloc[:lil_test_number, :-1]
  lil_testY = trainXY.iloc[:lil_test_number, -1]

  lil_train = trainXY.iloc[lil_test_number:, :-1]
  lil_trainY = trainXY.iloc[lil_test_number:, -1]

  # Select K Best ON LIL TRAIN
  skb = SelectKBest(k=10000).fit(lil_train, lil_trainY)
  mask = np.array(skb.get_support())
  lil_train = lil_train.iloc[:, mask]
  lil_test = lil_test.iloc[:, mask]

  size = len(lil_train.index)   


  # ## %% SVC train
  # clf = SVC(kernel='rbf', C=5)
  # # Fit all lil train
  # clf.fit(X, Y)

  # print("Train acc : %.3f"% clf.score(X,Y))
  # print("Test acc : %.3f"% clf.score(lil_test,lil_testY))

# %% Cross Validation
  clf = SVC(kernel='rbf', C=5)
  scores = []
  scores_test = []
  iterations = 10
  for i in range(iterations):
    startf = int(i*size/iterations)
    endf = int((i+1)*size/iterations)

    XY = lil_train 
    XY['label'] = pd.Series(lil_trainY, index=XY.index)
    XY = XY.sample(frac=1)#,random_state=i)

    X = XY.iloc[:,:-1]
    Y = XY.iloc[:,-1]

    vfx = X.iloc[startf:endf]
    vfy = Y[startf:endf]

    tfx = X.iloc[0:startf].append(X.iloc[endf:size])
    tfy = np.concatenate((Y[0:startf], Y[endf:size]))

    clf.fit(tfx, tfy)
    scores.append(clf.score(vfx, vfy))
    scores_test.append(clf.score(lil_test,lil_testY))


  cv_acc = sum(scores)/len(scores)
  print("Mean cross validation acc : %.3f"% cv_acc)
  # print(scores)
  cv_acc = sum(scores_test)/len(scores_test)
  print("Mean cross validation test acc : %.3f"% cv_acc)
  # print(scores_test)
  big_scores_test.append(cv_acc)

print("Total: %.3f"% (sum(big_scores_test)/len(big_scores_test)))


# # %% Deep Learning
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense

# model = keras.Sequential()

# model.add(Dense(
#     units=1000,
#     activation='relu',
#     kernel_initializer=keras.initializers.random_normal(),
#     bias_initializer=keras.initializers.random_normal()
# ))
# model.add(Dense(
#     units=100,
#     activation='relu',
#     kernel_initializer=keras.initializers.random_normal(),
#     bias_initializer=keras.initializers.random_normal()
# ))
# model.add(keras.layers.Dense(
#     units=1,
#     activation='sigmoid',
#     kernel_initializer=keras.initializers.random_normal(),
#     bias_initializer=keras.initializers.random_normal()
# ))
# model.compile(
#   optimizer=keras.optimizers.Adam(learning_rate=1e-5),
#   loss="binary_crossentropy",
#   metrics=['accuracy']
# )

# model.fit(
#   lil_train, 
#   lil_trainY, 
#   batch_size=32, 
#   epochs=10
# )
# _, acc = model.evaluate(lil_test, lil_testY, verbose=0)
# print("Acc : %.4f"%acc)



# %%

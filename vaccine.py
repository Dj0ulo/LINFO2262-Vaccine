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
real_test = test_data[:]

# Convert last 6 columns
def last_6(df):
  df.iloc[:, -6:] = df.iloc[:, -6:]\
      .apply(lambda col:
            col
            .apply(lambda v: str(v)[2:])
            .apply(lambda x: -1.0 if x == 'n' else float(x))
            )
last_6(train)
last_6(real_test)

# %% Filter out null variances
variances = np.array(train.var())
train_var = train.iloc[:, variances > 0]
real_test = real_test.iloc[:, variances > 0]

# %% Mean 
means = train_var.mean()
train_var = train_var.fillna(means)
real_test = real_test.fillna(means)
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

real_test_norm = pd.DataFrame(scaler.transform(real_test))
real_test_norm.columns = real_test_norm.columns

# %%Do every thing many times
big_scores_test = []
for big_it in range(100,110):
  print(big_it)
  # %% Randomize and clean data set
  trainXY = train_norm.loc[:]
  trainXY['label'] = pd.Series(trainY, index=trainXY.index)
  trainXY = trainXY.sample(frac=1,random_state=big_it) # I found that the best random_state was 104 with a train set of 100 rows

  lil_test_number = 100
  
  lil_test = trainXY.iloc[:lil_test_number, :-1]
  lil_testY = trainXY.iloc[:lil_test_number, -1]

  lil_train = trainXY.iloc[lil_test_number:, :-1]
  lil_trainY = trainXY.iloc[lil_test_number:, -1]


  # Select K Best ON LIL TRAIN
  skb = SelectKBest(k=10000).fit(lil_train, lil_trainY)
  mask = np.array(skb.get_support())
  lil_train = lil_train.iloc[:, mask]
  lil_test = lil_test.iloc[:, mask]
  real_test_clean = real_test_norm.iloc[:, mask]

  size = len(lil_train.index)

  lil_trainXY = lil_train.loc[:]
  lil_trainXY['label'] = pd.Series(lil_trainY, index=lil_trainXY.index)

## %% Cross Validation
  clf = SVC(kernel='rbf', C=5)

  scores = []
  scores_test = []
  iterations = 10
  test_portion = 0.1
  size_tf = int(test_portion*size)
  for i in range(iterations):
    XY = lil_trainXY.sample(frac=1,random_state=i)

    X = XY.iloc[:,:-1]
    Y = XY.iloc[:,-1]

    cross_text_x = X.iloc[:size_tf]
    cross_test_y = Y[:size_tf]

    cross_train_x = X.iloc[size_tf:]
    cross_train_y = Y[size_tf:]

    clf.fit(cross_train_x, cross_train_y)
    scores.append(clf.score(cross_text_x, cross_test_y))
    scores_test.append(clf.score(lil_test,lil_testY))


  cv_acc = sum(scores)/len(scores)
  print("Mean cross validation acc : %.3f"% cv_acc)
  # print(scores)
  cv_acc = sum(scores_test)/len(scores_test)
  print("Mean cross validation test acc : %.3f"% cv_acc)
  # print(scores_test)
  big_scores_test.append(cv_acc)

## %% Fit all lil train
  clf.fit(lil_train, lil_trainY)

  print("Train acc : %.3f"% clf.score(lil_train,lil_trainY))

  # I took this value for the BCR
  print("Test acc (BCR) : %.3f"% clf.score(lil_test,lil_testY))

  real_prediction = clf.predict(real_test_clean)

  pred = pd.DataFrame(real_prediction)
  pred = pred.apply(lambda col: col.apply(lambda x: int(x)))
  pred.to_csv('pred%d.csv'%big_it, quoting=csv.QUOTE_ALL)

print("Total: %.3f"% (sum(big_scores_test)/len(big_scores_test)))



# %% Deep Learning
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

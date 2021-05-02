# %% Imports
import pickle as pk
import math
import csv
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# %% Loading data
test_data = pk.load(open("test.pickle", "rb"))
(train_data_1, train_labels_1) = pk.load(open("train1.pickle", "rb"))
(train_data_2, train_labels_2) = pk.load(open("train2.pickle", "rb"))

# %% Merge train data
total_train = train_data_1
total_train = total_train.append(train_data_2)

# %% Sub-train
train = total_train[:]

# Convert last 6 columns
train.iloc[:, -6:] = train.iloc[:, -6:]\
    .apply(lambda col:
           col
           .apply(lambda v: str(v)[2:])
           .apply(lambda x: -1.0 if x == 'n' else float(x))
           )

# %% Mean 
means = train.mean()
train_no_nan = train.fillna(means)
# %% Variance features
var_filename = "train_variances.csv"

# selector = VarianceThreshold()
# selector.fit(train_no_nan)
# variances = np.asarray(selector.variances_)
# np.savetxt(var_filename, variances, delimiter=",")

with open(var_filename) as f:
    variances = list(map(pd.to_numeric, f.readlines()))



# %% Filter out null variances
train_var = train_no_nan.iloc[:, np.array(variances) > 0]
# %%

pca = PCA(n_components=2)
result = pca.fit_transform(train)

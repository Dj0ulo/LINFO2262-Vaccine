# %% Imports
import pickle as pk
import math
import csv
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
# %% Filter out null variances
variances = train_no_nan.var()
train_var = train_no_nan.iloc[:, np.array(variances) > 0]
# %% Save train
with open(r"train_nonan_var.pickle", "wb") as output_file:
  pk.dump(train_var, output_file)
# %% Load the first prefiltered train data
train_var = pk.load(open("train_nonan_var.pickle", "rb"))

# %%
scaler = StandardScaler()
train_norm = pd.DataFrame(scaler.fit_transform(train_var))
train_norm.columns = train_var.columns


# %%

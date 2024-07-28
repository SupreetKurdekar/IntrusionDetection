# %%
import json

import pandas as pd

traindf = pd.read_csv("KDDTrain.csv")
testdf = pd.read_csv("KDDTest.csv")

# %%
print(traindf[traindf.attack_class == 'normal'].shape[0] / len(traindf))
testdf[testdf.attack_class == 'normal'].shape[0] / len(testdf)

# %%
def classLabel(row):
  if row['attack_class'] == 'normal':
    return 0
  else:
    return 1

# %%
testdf['label'] = testdf.apply(classLabel,1)

traindf['label'] = traindf.apply(classLabel,1)

# add column for binary classification to both datasets

# %%

# %%
# traindf['duration'].plot()

# %%
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset

import torch

import numpy as np

from sklearn.preprocessing import OneHotEncoder

# %%
def get_encoder(traindf):
  categorical_columns = traindf.iloc[:,:-3].select_dtypes(include=['object']).columns.tolist()
  #Initialize OneHotEncoder
  encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
  encoder.fit(traindf[categorical_columns])
  return encoder,categorical_columns

def encode_df(encoder, df, categorical_columns):

  # Apply one-hot encoding to the categorical columns
  one_hot_encoded = encoder.transform(df[categorical_columns])

  #Create a DataFrame with the one-hot encoded columns
  #We use get_feature_names_out() to get the column names for the encoded data
  one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names(categorical_columns))

  colNanCounts = one_hot_df.isnull().sum().sum()

  # Concatenate the one-hot encoded dataframe with the original dataframe
  df_encoded = pd.concat([one_hot_df, df], axis=1)

  # Drop the original categorical columns
  df_encoded = df_encoded.drop(categorical_columns, axis=1)

  return df_encoded,colNanCounts

# %%
encoder, categorical_columns = get_encoder(traindf=traindf)

train_encoded,train_nan_counts = encode_df(encoder,traindf,categorical_columns)

# %%
test_encoded,test_nan_counts = encode_df(encoder,testdf,categorical_columns)

# %%
class NslDataSet(Dataset):
  def __init__(self,df):
    # self.input_df = df.iloc[:,:-3]
    # self.label_df = df.iloc[:,-1:]

    self.input_tensor = torch.tensor(df.iloc[:,:-3].values)
    self.label_tensor = torch.tensor(df.iloc[:,-1:].values)

    # print(self.input_tensor.shape)
    # print(self.label_tensor.shape)


  def __len__(self):
    return len(self.input_tensor)
  
  def __getitem__(self, index):
    
    return self.input_tensor[index],self.label_tensor[index]

# %%
trainDSet = NslDataSet(train_encoded)
testDSet = NslDataSet(test_encoded)

train_loader = DataLoader(trainDSet, batch_size=4,num_workers=0, shuffle=True)
test_loader = DataLoader(testDSet,batch_size=1,shuffle=False)

# k = iter(train_loader)

# %%
a,b = trainDSet.__getitem__(0)

print(a.shape)
print(b.shape)

# %%
trainIter = iter(train_loader)

# %%
trainSample,TrainLabel = next(trainIter)

# %%
print(trainSample)

# %%
# for batch_num, (row, label) in enumerate(train_loader):
  
#   if batch_num > 3:
#     break

# %%
# help(train_loader)



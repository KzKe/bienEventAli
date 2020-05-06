import numpy as np
import pandas as pd 
from config import Config
from sklearn import preprocessing
from nfold import CrossValidation


train_df = pd.read_csv(Config.TRAINING_FILE ,delimiter="\t", 
                                  names=["ID", "txt", "type", "company"])
dev_df = pd.read_csv(Config.DEV_FILE,delimiter="\t", 
                                  names=["ID", "txt", "type", "company"])

train_df = train_df.dropna()

le = preprocessing.LabelEncoder()
le.fit(train_df['type'])
types = le.transform(train_df['type'])
train_df['event_type'] = types

cv = CrossValidation(train_df, shuffle=True, target_cols=["type"], 
                      problem_type="multiclass_classification")
df_split = cv.split().reindex()

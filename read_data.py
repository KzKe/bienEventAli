import numpy as np
import pandas as pd 
from config import Config

train_df = pd.read_csv(Config.TRAINING_FILE ,delimiter="\t", 
                                  names=["ID", "txt", "type", "company"])

dev_df = pd.read_csv(Config.DEV_FILE,delimiter="\t", 
                                  names=["ID", "txt", "type", "company"])



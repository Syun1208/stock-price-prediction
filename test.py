import tensorflow
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path


def test():
    pass


if __name__ == '__main__':
    df_test = pd.read_csv('test/money/BTC-USD-test.csv')
    df_test = df_test.to_dict('list')
    scale = MinMaxScaler(feature_range=(0, 1))
    df_test = scale.fit_transform(np.array(df_test['Close']).reshape(-1, 1))
    print(np.array(df_test).shape)

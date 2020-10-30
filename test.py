import torch
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pickle
from collections import deque

from functools import partial

def sliding_window(seq, n=4):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

def test():
    # for i, j in enumerate(x):
    #     print(1)
    # key_column = "store_item_id"
    # additional_columns = ["store", "item", "data", "yearly_corr"]
    # x = list(set([key_column] + additional_columns))
    # print(x)

    df = pd.DataFrame()
    y = [np.random.uniform(0, 20, size=(2, 2)) for _ in range(6)]
    df['test_col'] = y
    print(df.head())

    x = list(list(x) for x in sliding_window(y))

    # x = sliding_window(x, n=2)
    print(x)
    print(len(x))
    print(np.shape(x[0]))




    # def add(x, a, b):
    #     return x + 10 * a + 100 * b
    #
    # add = partial(add, a=2)
    #
    # print(add(x=4, b=1))
    # train = pd.read_csv('./data/train.csv')
    # test = pd.read_csv('./data/test.csv')
    #
    # train['date'] = pd.to_datetime(train['date'])
    # test['date'] = pd.to_datetime(test['date'])
    #
    # test['sales'] = np.nan
    # data = pd.concat([train, test], ignore_index=True)
    # data['store_item_id'] = data['store'].astype(str) + '_' + data['item'].astype(str)
    #
    # data['dayofweek'] = data['date'].dt.dayofweek
    # data['month'] = data['date'].dt.month
    # data['year'] = data['date'].dt.year
    # data['day'] = data['date'].dt.day
    #
    # data['dayofweek_sin'] = sin_transform(data['dayofweek'])
    # data['dayofweek_cos'] = cos_transform(data['dayofweek'])
    # data['month_sin'] = sin_transform(data['month'])
    # data['month_cos'] = cos_transform(data['month'])
    # data['day_sin'] = sin_transform(data['day'])
    # data['day_cos'] = cos_transform(data['day'])
    #
    # data.drop('id', axis=1, inplace=True)
    #
    # data = data.sort_values(['store_item_id', 'date'])
    #
    # train['store_item_id'] = train['store'].astype(str) + '_' + train['item'].astype(str)
    #
    #
    # mode = 'valid'
    # if mode == 'valid':
    #     scale_data = train[train['date'] < '2017-01-01']
    # else:
    #     scale_data = train[train['date'] >= '2014-01-01']
    #
    # scale_map = {}
    # scaled_data = pd.DataFrame()
    # i = 0
    # for store_item_id, item_data in data.groupby('store_item_id', as_index=False):
    #     print('new')
    #     print(item_data.head())
    #     sidata = scale_data.loc[scale_data['store_item_id'] == store_item_id, 'sales']
    #     print(sidata.head())
    #     mu = sidata.mean()
    #     sigma = sidata.std()
    #     yearly_autocorr = get_yearly_autocorr(sidata)
    #     print("item data what?")
    #     print(item_data.loc[:, 'sales'].head())
    #     item_data.loc[:, 'sales'] = (item_data['sales'] - mu) / sigma
    #     print(item_data.head())
    #     scale_map[store_item_id] = {'mu': mu, 'sigma': sigma}
    #     item_data['mean_sales'] = mu
    #     item_data['yearly_corr'] = yearly_autocorr
    #     scaled_data = pd.concat([scaled_data, item_data], ignore_index=True)
    #     i += 1
    #     if i == 2:
    #         print(scaled_data.head())
    #         print(scaled_data.tail())
    #         break
    #     else:
    #         print(scaled_data.head())
    #         print(scaled_data.tail())
    #         continue


def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))

def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    print(ac)
    print(ac[365])
    print(ac[364])
    print(ac[366])
    x = (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])
    print(x)
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])

if __name__ == '__main__':
    test()

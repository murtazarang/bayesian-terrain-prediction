import numpy as np
import pickle

import pandas as pd
import gc

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf

DAYS_IN_A_YEAR = 366  # Account for leap year, instead of discarding it.


# Read data from file and process only the relevant information
def load_data(args):
    if args.model in ["1D", "1d"]:
        args.num_features = 1
        fert_df = pd.read_csv("./data/Fertilizer1dAnnual.csv")

    elif args.model in ["3D", "3d"]:
        if args.xdim * args.ydim != args.num_features:
            assert "Incorrect Feature Dimensions"
        fert_df = pd.read_csv("./data/Fertilizer3dAnnual.csv")

    else:
        assert "Invalid Input Model Selected"
        return

    fert_df["date"] = pd.to_datetime(fert_df["date"])
    # Drop the single day of 2010
    fert_df.drop(fert_df[fert_df.date == '2010-01-01'].index, inplace=True)

    # Alleviate exponential effects, transform the target variable with log-transformation
    # https://arxiv.org/abs/1709.01907
    height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    for i, h in enumerate(height_list):
        fert_df['log_h' + str(i + 1)] = np.log(fert_df[h])
        # Compress it since it blows up the dataframe size
        c_min = fert_df['log_h' + str(i + 1)].min()
        c_max = fert_df['log_h' + str(i + 1)].max()
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            fert_df['log_h' + str(i + 1)] = fert_df['log_h' + str(i + 1)].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            fert_df['log_h' + str(i + 1)] = fert_df['log_h' + str(i + 1)].astype(np.float32)
        else:
            fert_df['log_h' + str(i + 1)] = fert_df['log_h' + str(i + 1)].astype(np.float64)

    # log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]

    # Specify cyclical nature of the data per year with sine/cosine date values
    # https: // ianlondon.github.io / blog / encoding - cyclical - features - 24hour - time /
    fert_df['day_of_year_sin'] = np.sin(2 * np.pi * fert_df["day_of_year"] / DAYS_IN_A_YEAR)
    fert_df['day_of_year_cos'] = np.cos(2 * np.pi * fert_df["day_of_year"] / DAYS_IN_A_YEAR)

    # Normalize the year
    fert_df["year_mod"] = (fert_df['year'] - fert_df['year'].min()) / (fert_df['year'].max() - fert_df['year'].min())

    # Split the data for training, and testing before computing the correlation
    train_data, test_data = split_timeseries_data(args, fert_df)

    # Now we perform scaling/normalization on the training and omit the validation/target set.
    scale_data_train = train_data[train_data['date'] < args.testing_start_date]
    scale_data_test = test_data

    """
    Scaling Train and Test Data Independently
    """
    scale_map_train = {}
    scaled_data_train = pd.DataFrame()
    scaled_data_train = pd.concat([scaled_data_train, scale_data_train], ignore_index=True)

    scale_map_test = {}
    scaled_data_test = pd.DataFrame()
    scaled_data_test = pd.concat([scaled_data_test, scale_data_test], ignore_index=True)

    # To capture the yearly trend of the fertilizer height we also standardize and compute the yearly
    # auto-correlation for each height.
    for h in height_list:
        h_i_data = scale_data_train[h]
        h_i_mean = h_i_data.mean()
        h_i_var = h_i_data.std()
        # print(f"Training_{h}")
        h_year_autocorr = get_yearly_autocorr(h_i_data)
        # Standardize for Train
        scaled_data_train[h + '_yearly_corr'] = h_year_autocorr
        scaled_data_train[h] = (scaled_data_train[h] - h_i_mean) / h_i_var
        scale_map_train[h] = {'mu': h_i_mean, 'sigma': h_i_var}

        # Do the same for Test data
        h_i_data = scale_data_test[h]
        h_i_mean = h_i_data.mean()
        h_i_var = h_i_data.std()
        # print(f"Testing_{h}")
        h_year_autocorr = get_yearly_autocorr(h_i_data)
        # print('\n')
        # Standardize for Test
        scaled_data_test[h + '_yearly_corr'] = h_year_autocorr
        scaled_data_test[h] = (scaled_data_test[h] - h_i_mean) / h_i_var
        scale_map_test[h] = {'mu': h_i_mean, 'sigma': h_i_var}

    # yearly_autocorr_height_list = [h + '_yearly_corr' for h in height_list]
    # selected_columns = ['date', 'day_of_year', 'year', 'day_of_year_sin', 'day_of_year_cos', 'year_mod'] + height_list \
    #                     + log_height_list + yearly_autocorr_height_list

    # Drop unnecessary features from Train and Test Data
    if args.model in ['1D', '1d']:
        scaled_data_train.drop(['day_of_year', 'year', 'drymatter', 'heightchange', 'cover'], axis=1, inplace=True)
        scaled_data_test.drop(['day_of_year', 'year', 'drymatter', 'heightchange', 'cover'], axis=1, inplace=True)
    elif args.model in ['3D', '3d']:
        scaled_data_train.drop(['day_of_year', 'year'], axis=1, inplace=True)
        scaled_data_test.drop(['day_of_year', 'year'], axis=1, inplace=True)

    print("General raw metrics for height for (x_0, y_0): ")
    print(fert_df['h1'].describe())
    print('\n')
    # print("Input data: ")
    # print(scaled_data.head())
    # print('\n')
    # print("Input Features")
    # if args.model in ["1d", "1D"]:
    #     print(scaled_data.drop(['date', 'day_of_year', 'year', 'drymatter', 'heightchange', 'cover'], axis=1).head())
    # elif args.model in ["3D", "3d"]:
    #     print(scaled_data.drop(['date', 'day_of_year', 'year'], axis=1).head())
    # print('\n')
    if args.compress_data:
        print("Compressing Data")
        scaled_data_train = reduce_mem_usage(scaled_data_train)
        scaled_data_test = reduce_mem_usage(scaled_data_test)

    scaled_data_train.to_pickle('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
    scaled_data_test.to_pickle('./data/processed_data/' + args.model + '_test_processed_data' + '.pkl')
    pickle.dump(scale_map_train, open('./data/processed_data/' + args.model + '_scale_map_train.pkl', 'wb'))
    pickle.dump(scale_map_test, open('./data/processed_data/' + args.model + '_scale_map_test.pkl', 'wb'))


def split_timeseries_data(args, data):
    # Let's split the data into the following parts
    # Train: 1980-01-01 ~ 2003-12-31
    # Validation: 2004-01-01 ~ 2007-12-31
    # Test: 2008-01-01 ~ 2009-12-31
    print("Timeline of input data: ")
    print(data['date'].min(), " to ", data['date'].max())
    train_data = data[data['date'] < args.testing_start_date]
    test_data = data[(data['date'] >= args.testing_start_date) & (data['date'] <= args.testing_end_date)]

    print("Train Data Timeline: ")
    print(train_data['date'].min(), " to ", train_data['date'].max())
    print("Test Data Timeline: ")
    print(test_data['date'].min(), " to ", test_data['date'].max())
    print('\n')

    return train_data, test_data


def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    # print(np.shape(ac))
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])


def last_year_lag(col):
    return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
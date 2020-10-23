import numpy as np
import pickle

import pandas as pd
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf

DAYS_IN_A_YEAR = 366  # Account for leap year, instead of discarding it.


# Read data from file and process only the relevant information
def load_data(args):
    if args.model in ["1D", "1d"]:
        args.num_features = 1
        fert_df = pd.read_csv("./data/Fertilizer1dAnnual.csv")

    elif args.model in ["3D", "3d"]:
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

    log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]

    # Specify cyclical nature of the data per year with sine/cosine date values
    # https: // ianlondon.github.io / blog / encoding - cyclical - features - 24hour - time /
    fert_df['day_of_year_sin'] = np.sin(2 * np.pi * fert_df["day_of_year"] / DAYS_IN_A_YEAR)
    fert_df['day_of_year_cos'] = np.cos(2 * np.pi * fert_df["day_of_year"] / DAYS_IN_A_YEAR)

    # Normalize the year
    fert_df["year_mod"] = (fert_df['year'] - fert_df['year'].min()) / (fert_df['year'].max() - fert_df['year'].min())

    # Split the data for training, and testing before computing the correlation
    train_data, test_data = split_timeseries_data(args, fert_df)

    # Now we perform scaling/normalization on the training and omit the validation/target set.
    if args.training_mode == 'train':
        scale_data = train_data[train_data['date'] < args.validation_start_date]
    elif args.training_mode == 'test':
        scale_data = train_data[train_data['date'] < args.training_start_date]

    # To capture the yearly trend of the fertilizer height we also standardize and compute the yearly autocorrelation for each height.
    scale_map = {}
    scaled_data = pd.DataFrame()
    scaled_data = pd.concat([scaled_data, scale_data], ignore_index=True)

    for h in height_list:
        h_i_data = scale_data[h]
        h_i_mean = h_i_data.mean()
        h_i_var = h_i_data.std()
        h_year_autocorr = get_yearly_autocorr(h_i_data)
        # Standardize
        scaled_data[h + '_yearly_corr'] = h_year_autocorr
        scaled_data[h] = (scaled_data[h] - h_i_mean) / h_i_var
        scale_map[h] = {'mu': h_i_mean, 'sigma': h_i_var}

    yearly_autocorr_height_list = [h + '_yearly_corr' for h in height_list]

    selected_columns = ['date', 'day_of_year', 'year', 'day_of_year_sin', 'day_of_year_cos', 'year_mod'] + height_list\
                       + log_height_list + yearly_autocorr_height_list

    print("General raw metrics for height for (x_0, y_0): ")
    print(fert_df['h1'].describe())
    print('\n')
    print("Input data: ")
    print(scaled_data.head())
    print('\n')

    print("Input Features")
    if args.model in ["1d", "1D"]:
        print(scaled_data.drop(['date', 'day_of_year', 'year', 'drymatter', 'heightchange', 'cover'], axis=1).head())
    elif args.model in ["3D", "3d"]:
        print(scaled_data.drop(['date', 'day_of_year', 'year'], axis=1).head())
    print('\n')

    if args.model in ['1D', '1d']:
        scaled_data.to_pickle('./data/processed_data/' + args.dataset + 'pkl')
    elif args.model in ['3D', '3d']:
        scaled_data.to_pickle('./data/processed_data/' + args.dataset + '.pkl')


def split_timeseries_data(args, data):
    # Let's split the data into the following parts
    # Train: 1980-01-01 ~ 2003-12-31
    # Validation: 2004-01-01 ~ 2007-12-31
    # Test: 2008-01-01 ~ 2009-12-31
    print("Timeline of input data: ")
    print(data['date'].min(), " to ", data['date'].max())
    if args.training_mode == "train":
        train_data = data[data['date'] < args.testing_start_date]
        test_data = data[(data['date'] >= args.testing_start_date) & (data['date'] <= args.testing_end_date)]

    elif args.training_mode == "test":
        train_data = data[(data['date'] >= '1980-01-01') & (data['date'] < args.testing_start_date)]
        test_data = data[(data['date'] >= args.testing_start_date) & (data['date'] <= args.testing_end_date)]
    else:
        assert "Incorrect training mode selected."
        return

    print("Train Data Timeline: ")
    print(train_data['date'].min(), " to ", train_data['date'].max())
    print("Test Data Timeline: ")
    print(test_data['date'].min(), " to ", test_data['date'].max())
    print('\n')

    return train_data, test_data


def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])
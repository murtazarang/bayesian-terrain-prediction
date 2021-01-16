import plotly.graph_objects as go
import pandas as pd
import scipy.io
import numpy as np
import csv
# Read data from a csv

def plot_surface(z_data, title):
    z = z_data
    # print(z.shape)
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=title, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

# Reshape to feed to Matlab
def np_to_csv(y_pred_mean, y_pred_std, y_target, y_date, args):
    # Flatten the 3D data, and unroll over the whole sequence
    # seq_len = y_pred_mean.shape[1]
    print(f'pred: {y_pred_mean.shape}, target: {y_target.shape}, date: {y_date.shape}')
    # y_pred_mean = y_pred_mean.reshape(y_pred_mean.shape[0], y_pred_mean.shape[1], -1)
    # y_pred_mean = y_pred_mean.reshape(y_pred_mean.shape[0], -1)

    # y_pred_std = y_pred_std.reshape(y_pred_std.shape[0], y_pred_std.shape[1], -1)
    # y_pred_std = y_pred_std.reshape(y_pred_std.shape[0], -1)

    # y_target = y_target.reshape(y_target.shape[0], y_target.shape[1], -1)
    # y_target = y_target.reshape(y_target.shape[0], -1)

    # y_final = np.concatenate((y_date, y_pred_mean, y_pred_std, y_target), axis=-1)

    # y_pred_f_t = []
    # y_pred_std_f_t = []
    # y_target_f_t = []
    #
    # for t in range(seq_len):
    #     y_pred_f_t += ['h_pred_mean_' + str(f) + '_' + str(t) for f in range(args.num_features)]
    #     y_pred_std_f_t += ['h_pred_std_' + str(f) + '_' + str(t) for f in range(args.num_features)]
    #     y_target_f_t += ['h_target_' + str(f) + '_' + str(t)for f in range(args.num_features)]

    # pred_features = ['date'] + y_pred_f_t + y_pred_std_f_t + y_target_f_t

    y_mdic = {'date': y_date, 'y_pred_mean': y_pred_mean, 'y_pred_std': y_pred_std, 'y_target': y_target}
    scipy.io.savemat('./data/prediction_data/' + args.model + '_predict_data_' + args.predict_run + '.mat', mdict=y_mdic, oned_as='row')

    # matdata = scipy.io.loadmat('./data/prediction_data/' + args.model + '_predict_data_' + args.predict_run + '.mat')
    # print(matdata.keys())
    #
    # print(matdata['y_pred_mean'].shape)
    # print(matdata['y_pred_mean'][0][0][0])
    # print(matdata['y_pred_std'].shape)
    # print(matdata['y_target'].shape)
    # print(matdata['date'].shape)

    # with open('./data/prediction_data/' + args.model + '_predict_data_' + args.predict_run + '.csv', 'w', newline='', encoding='utf-8') as pred_csv:
    #     csvWriter = csv.writer(pred_csv, delimiter=',', lineterminator='\n')
    #     csvWriter.writerow(pred_features)
    #     csvWriter.writerows(y_final)
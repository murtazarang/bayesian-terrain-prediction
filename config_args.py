import argparse


def parse_args():
    parser = argparse.ArgumentParser("TimeSeries")

    parser.add_argument("--model", type=str, default="1D", choices=["1D", "1d", "3D", "3d"])
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--xdim", type=int, default=10)
    parser.add_argument("--ydim", type=int, default=10)

    parser.add_argument("--in_seq_len", type=int, default=60)
    parser.add_argument("--out_seq_len", type=int, default=30)

    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--train_split", type=int, default=8258)

    # 1D Bayesian LSTM Architecture
    parser.add_argument("--lstm1_dim", type=int, default=128)
    parser.add_argument("--lstm2_dim", type=int, default=32)

    # 2D ConvLSTM Architecture
    parser.add_argument("--conv1_dim", type=int, default=10)
    parser.add_argument("--conv2_dim", type=int, default=5)
    parser.add_argument("--conv3_dim", type=int, default=1)

    # Bayesian Uncertainity Parameters
    parser.add_argument("--bayes-dropout", type=float, default=0.5)

    # Data pre-processing methods
    parser.add_argument("--no_cylical_dates", action='store_false')
    parser.add_argument("--height_correction", type=str, default='log', choices=['log', 'autocorrelation'])
    parser.add_argument("--training_mode", type=str, default="train", choices=["test", "train"])
    parser.add_argument("--testing_start_date", type=str, default='2009-01-01')
    parser.add_argument("--testing_end_date", type=str, default='2009-12-31')
    parser.add_argument("--validation_start_date", type=str, default='2003-01-01')

    # Data Processing Requirements
    parser.add_argument("--load_data", action='store_true')
    parser.add_argument("--sequence_data", action='store_true')
    parser.add_argument("--compress_data", action='store_true')
    parser.add_argument("--dataset", type=str, default='_Fertilizer1dAnnual')

    return parser.parse_args()


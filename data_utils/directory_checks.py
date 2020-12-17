import os
import sys
from pathlib import Path


def dir_checks(args):
    process_dir = Path('./data/processed_data')
    if not process_dir.exists():
        os.makedirs(process_dir)

    seq_dir = Path('./data/sequence_data/numpy/')
    if not seq_dir.exists():
        os.makedirs(seq_dir)

    pred_dir = Path('./data/prediction_data')
    if not seq_dir.exists():
        os.makedirs(pred_dir)

    if args.training_mode in ['train_init', 'train_final']:
        if args.load_data:
            train_file = Path('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
            test_file = Path('./data/processed_data/' + args.model + '_test_processed_data' + '.pkl')

            if train_file.exists():
                print("Train Processed Pickle Exists")
                sys.exit()

            if test_file.exists():
                print("Test Processed Pickle Exists")
                sys.exit()

        if args.sequence_data:
            train_file = Path('./data/sequence_data/' + args.model + '_seq_data_' + str(args.in_seq_len) + '_' +
                              str(args.out_seq_len) + '.pkl')

            if train_file.exists():
                print("Seq Pickle Exists")
                sys.exit()

        if args.sequence_to_np:
            train_file = Path(
                './data/sequence_data/numpy/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
                str(args.out_seq_len) + '.h5')
            test_file = Path(
                './data/sequence_data/numpy/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                str(args.out_seq_len) + '.h5')

            if train_file.exists():
                print("Train File NP Seq Exists")
                sys.exit()

            if test_file.exists():
                print("Test File NP Seq Exists")
                sys.exit()


    if args.predict:
        predict_file = Path('./data/prediction_data/' + args.model + '_predict_data_' + args.predict_run + '.csv')

        if predict_file.exists():
            print("Prediction Run CSV File Exists")
            sys.exit()

    return

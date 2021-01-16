# Agriculture Height Prediction with ConvLSTM


## Dependencies
* [Install and Setup Anaconda](https://www.anaconda.com/products/individual)

* Clone this repository and `cd` into root folder of the project.

* Create a new conda environment using the dependency file in the project folder.

```
bash
conda env create -f environment.yml
```

* Activate the environment to run the training or testing scripts.
```
bash
conda activate ag-bay-torch
```

## Arguments
To Run the predictions for the first time, or using a new dataset please ensure
that you have the following arguments enabled:

* `--load_data`: Scales and performs necessary pre-processing of the input data for creating final
sliding window inputs for the network.

* `--sequence_data`: Create sliding windows and stores into a **csv** file

* `--sequence_to_np`: Generates numpy arrays and stores in an hdf5 format, for read on access memory, which can be used directly in the Pytorch DataLoader function.

* `--in_seq_len VALUE`: Input Sequence Length. Expects an **int** value.
* `--out_seq_len VALUE`: Output Sequence Length. Expects an **int** value.

* `--device cpu`: To run inference or training on CPU, else default is at `cuda`.

* `--exp_name MODEL_NAME`: Experiment name to load the model during inference, or save the mode as during training.

## Prediction/Testing
```
bash
python main.py --in_seq_len IN_VALUE --out_seq_len OUT_VALUE --load_data --sequence_data --sequence_to_np
```

## Training
```
bash
python main.py --train_network --in_seq_len IN_VALUE --out_seq_len OUT_VALUE --load_data --sequence_data --sequence_to_np
```

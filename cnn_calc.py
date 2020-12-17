def find_settings(shape_in, shape_out, kernel_sizes, dilation_sizes, padding_sizes, stride_sizes, transpose=False):
    from itertools import product

    import torch
    from torch import nn

    import numpy as np

    # Fake input
    x_in = torch.tensor(np.random.randn(4, 1, shape_in, shape_in), dtype=torch.float)

    # Grid search through all combinations
    for kernel, dilation, padding, stride in product(kernel_sizes, dilation_sizes, padding_sizes, stride_sizes):
        # Define a layer
        if transpose:
            layer = nn.ConvTranspose2d
        else:
            layer = nn.Conv2d
        layer = layer(
                1, 1,
                (4, kernel),
                stride=(2, stride),
                padding=(2, padding),
                dilation=(2, dilation)
            )

        # Check if layer is valid for given input shape
        try:
            x_out = layer(x_in)
        except Exception:
            continue

        # Check for shape of out tensor
        result = x_out.shape[-1]

        if shape_out == result:
            print('Correct shape for:\n ker: {}\n dil: {}\n pad: {}\n str: {}\n'.format(kernel, dilation, padding, stride))


def main():
    transpose = True
    shape_in = 7
    shape_out = 100

    kernel_sizes = [3, 5, 7, 9, 11]
    dilation_sizes = list(range(1, 20))
    padding_sizes = list(range(15))
    stride_sizes = list(range(0, 16))
    find_settings(shape_in, shape_out, kernel_sizes, dilation_sizes, padding_sizes, stride_sizes, transpose)



if __name__ == '__main__':
    main()
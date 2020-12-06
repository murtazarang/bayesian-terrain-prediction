from torch.utils.data import Dataset, IterableDataset, DataLoader


class CustomIterableDatasetv1(IterableDataset):

    def __init__(self, filename):
        # Store the filename in object's memory
        self.filename = filename

        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self, text):
        ### Do something with text here

        ###
        return text

    def line_mapper(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        # text, label = line.split(',')
        text = self.preprocess(line)

        return text

    def __iter__(self):
        # Create an iterator
        file_itr = open(self.filename)

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr


def main():
    dataset = CustomIterableDatasetv1('./data/test.csv')
    dataloader = DataLoader(dataset, batch_size=64)

    x = next(iter(dataloader))
    print(x)

    for X in dataloader:
        print(len(X))  # 64
        # print(y.shape)  # (64,)

        ### Do something with X and y

        ###


if __name__ == '__main__':
    main()


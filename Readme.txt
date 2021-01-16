Install Anaconda: https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

Then go to Downloads folder ->
bash ~/Downloads/Anaconda3-2020.11-Linux-x86_64.sh

Follow the instructions to install it.


Download my code from here: https://drive.google.com/drive/folders/1AdIjOcJy0PkglizKKy_KU7Hm7eyB5qTb?usp=sharing

Then copy it to a folder.

In the terminal "cd" to the root folder of the project code, where the "environment.yml" file is located.

Then install my environment with the code below in the terminal.
conda env create -f environment.yml



Activate my environment:
conda activate bayesian-torch


Copy the main csv file to the data folder and name it as 'Fertilizer3dAnnual.csv'

Then you can run my code as:

To Train:
python main.py --train_network --in_seq_len NUMBER --out_seq_len NUMBER --load_data --sequence_data -- sequence_to_np


Just to test
python main.py --in_seq_len NUMBER --out_seq_len NUMBER --load_data --sequence_data -- sequence_to_np




import pandas as pd
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from getData import getAALabelAndLengthAndEvolutionOnehot
from utils.HGNN import generate_G_from_H


class Dataset(object):

    def __init__(self, train_data_path, test_data_path, filenames, amino_acids, args):

        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.filenames = filenames
        self.amino_acids = amino_acids

        self.train_loader, self.test_loader, self.class_freq, self.train_num, self.label_encode, self.G = self.load_dataset(train_data_path, test_data_path)

    def load_dataset(self, train_data_path, test_data_path):
        train_seqvec, train_evolution, train_onehot, train_length, train_label, train_struct, train_name = \
            getAALabelAndLengthAndEvolutionOnehot(train_data_path, self.max_length, self.filenames)
        test_seqvec, test_evolution, test_onehot, test_length, test_label, test_struct, test_name = \
            getAALabelAndLengthAndEvolutionOnehot(test_data_path, self.max_length, self.filenames)

        label_encode = np.load(f"data/MFTP_label.npy")
        H = train_label.T
        G = generate_G_from_H(H)

        label_encode = torch.Tensor(label_encode)
        G = torch.Tensor(G)

        class_freq, train_num = self.get_class_freq(train_label)

        # 根据 shuffle 参数来决定是否在每个 epoch 前打乱数据
        train_loader = self.encode_data(train_seqvec, train_evolution,
                                        train_onehot, train_length,
                                        train_label, train_struct, shuffle=True)

        test_loader = self.encode_data(test_seqvec, test_evolution,
                                       test_onehot, test_length,
                                       test_label, test_struct, shuffle=False)

        return train_loader, test_loader, class_freq, train_num, label_encode, G

    def get_class_freq(self, labels):
        class_freq = np.sum(labels, axis=0)
        train_num = len(labels)

        return class_freq, train_num

    def encode_data(self, seqvec, evolution, onehot, length, label, struct, shuffle):

        seqvec = torch.Tensor(seqvec)
        evolution = torch.Tensor(evolution)
        onehot = torch.LongTensor(onehot)
        length = torch.Tensor(length)
        label = torch.Tensor(label)
        struct = torch.Tensor(struct)

        dataset = list(zip(seqvec, evolution, onehot, length, label, struct))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True)

        return loader


if __name__ == '__main__':
    filenames = ['anti-angiogenic peptide,', 'anti-bacterial peptide', 'anti-cancer peptide',
                 'anti-coronavirus peptide', 'anti-diabetic peptide', 'anti-endotoxin peptide',
                 'anti-fungal peptide', 'anti-HIV peptide', 'anti-hypertensive peptide',
                 'anti-inflammatory peptide', 'anti-MRSA peptide', 'anti-parasitic peptide',
                 'anti-tubercular peptide', 'anti   -viral peptide', 'blood-brain barrier peptide',
                 'biofilm-inhibitory peptide', 'cell-penetrating peptide', 'dipeptidyl peptidase IV peptide',
                 'quorum-sensing peptide', 'surface-binding peptide', 'tumor homing peptide'
                 ]

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                   'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=192)
    args = parser.parse_args()

    print("Dataset..........")
    dataset = Dataset(train_data_path=f"data/MFTP/Seqvec/train",
                      test_data_path=f"data/MFTP/Seqvec/test",
                      filenames=filenames,
                      amino_acids=amino_acids,
                      args=args)

    train_loader, test_loader = dataset.train_loader, dataset.test_loader


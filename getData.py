import json
import pickle
import numpy as np


def PadEncode_onehot(data, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, temp = [], []
    for i in range(len(data)):
        data[i] = [char for char in data[i].tolist()]
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)

        if length <= max_len:
            temp.append(elemt)
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
    return np.array(data_e)


def BLOSUM62_embedding(seq, max_len=50):
    f=open('data/blosum62.txt')
    text=f.read()
    f.close()
    text=text.split('\n')
    while '' in text:
        text.remove('')
    cha=text[0].split(' ')
    while '' in cha:
        cha.remove('')
    index=[]
    for i in range(1,len(text)):
        temp=text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j]=float(temp[j])
        index.append(temp)
    index=np.array(index)
    BLOSUM62_dict={}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]]=index[:,j]
    all_embeddings=[]
    for each_seq in seq:
        temp_embeddings=[]
        seq_list = [char for char in each_seq.tolist()]
        for each_char in seq_list:
            temp_embeddings.append(BLOSUM62_dict[each_char])
        if max_len>len(seq_list):
            zero_padding=np.zeros((max_len-len(seq_list),23))
            data_pad=np.vstack((temp_embeddings,zero_padding))
        elif max_len==len(seq_list):
            data_pad=temp_embeddings
        else:
            data_pad=temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings=np.array(all_embeddings)
    return all_embeddings


def getStruct(data):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    struct = np.zeros((len(data), len(amino_acids)), dtype=int)
    for i, peptide in enumerate(data):
        peptide = [char for char in peptide.tolist()]
        for aa in peptide:
            if aa in amino_acids:
                col_idx = amino_acids.index(aa)
                struct[i, col_idx] = 1
    return struct


def getAALabelAndLengthAndEvolutionOnehot(file_name, max_length, filenames):
    path_data = "{}.npz".format(file_name)
    path_label_name = "{}.json".format(file_name)
    path_label = "{}.fasta".format(file_name)

    # 获取数据
    data = np.load(path_data)

    # 获取标签
    with open(path_label_name) as file:
        labels = json.load(file)

    data_label = {}
    data_length = {}
    data_seqs = {}

    with open(path_label) as file:
        for each in file:
            each = each.strip()
            if each[0] == '>':
                pipe_index = each.index('|')
                index = each.index(',')
                values_before_pipe = each[1:pipe_index]
                values_inter_pipe = each[pipe_index + 1:index]
                values_after_pipe = each[index + 1:]

                data_label[str(values_before_pipe)] = np.array(list(values_inter_pipe), dtype=int)
                data_length[str(values_before_pipe)] = np.array(values_after_pipe, dtype=int)
            else:
                data_seqs[str(values_before_pipe)] = np.array(each, dtype=str)

    label = []
    length = []
    seqs = []
    seq_name = []
    for i in range(len(labels)):
        now_label = data_label[labels[i]]
        now_length = data_length[labels[i]]
        now_seqs = data[labels[i]]
        now_seq_name = data_seqs[labels[i]]

        padded_array = np.zeros((max_length, now_seqs.shape[1]))
        padded_array[:now_seqs.shape[0], :] = now_seqs

        label.append(now_label)
        length.append(now_length)
        seqs.append(padded_array)
        seq_name.append(now_seq_name)

    struct = getStruct(seq_name)
    evolution = BLOSUM62_embedding(seq_name, max_length)
    seqs_onehot = PadEncode_onehot(seq_name, max_length)

    seq_name = [''.join(sublist) for sublist in seq_name]
    # seq_name = pad_sequences(seq_name, max_length)

    return np.array(seqs), np.array(evolution), np.array(seqs_onehot), np.array(length), np.array(label), np.array(struct), seq_name


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + 'X' * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences


def get_label_num(dataset):
    with open(f"../data/{dataset}/label_dict.pkl", "rb") as file:
        label_dict = pickle.load(file)
        return len(label_dict), label_dict


if __name__ == '__main__':
    train_data_path = "data/MFTP/SeqVec/train"
    test_data_path = "data/MFTP/SeqVec/test"
    max_length = 50

    filenames = ['anti-angiogenic peptide,', 'anti-bacterial peptide', 'anti-cancer peptide',
                 'anti-coronavirus peptide', 'anti-diabetic peptide', 'anti-endotoxin peptide',
                 'anti-fungal peptide', 'anti-HIV peptide', 'anti-hypertensive peptide',
                 'anti-inflammatory peptide', 'anti-MRSA peptide', 'anti-parasitic peptide',
                 'anti-tubercular peptide', 'anti-viral peptide', 'blood-brain barrier peptide',
                 'biofilm-inhibitory peptide', 'cell-penetrating peptide', 'dipeptidyl peptidase IV peptide',
                 'quorum-sensing peptide', 'surface-binding peptide', 'tumor homing peptide'
                 ]

    train_seqvec, train_evolution, train_onehot, train_length, train_label, train_struct, train_name = \
        getAALabelAndLengthAndEvolutionOnehot(train_data_path, max_length, filenames)

    # print("train_seqvec的形状：", train_seqvec.shape)
    print("train_evolution的形状：", train_evolution.shape)
    # print("train_label的形状：", train_label.shape)

    # MFTP_label = np.load(f"data/MFTP_label.npy")
    # print(MFTP_label)
    # print(MFTP_label.shape)
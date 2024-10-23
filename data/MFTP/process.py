import numpy as np


def getSequenceData(file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}.txt".format(file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(each[1:])
            else:
                data.append(each)

    return data, label


def PadEncode(data, label):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'

    data_e, label_e, seq_length = [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        st = data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            sign = 0

        if sign == 0:
            seq_length.append(len(st))
            b += 1
            data_e.append(st)
            label_e.append(label[i])
    return data_e, label_e, seq_length


def writeFasta(file_name, data_e, label_e, length_e):
    out_path = "{}.fasta".format(file_name)

    with open(out_path, 'w') as file:
        # 逐行写入FASTA格式数据
        for i, (sequence, label, length) in enumerate(zip(data_e, label_e, length_e)):
            file.write(f'>{file_name}{i}|{label},{length}\n{sequence}\n')

    file.close()


def staticTrainAndTest(y_train, y_test):
    filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                 'AVP',
                 'BBP', 'BIP',
                 'CPP', 'DPPIP',
                 'QSP', 'SBP', 'THP']
    data_size_tr = np.zeros(len(filenames))
    data_size_te = np.zeros(len(filenames))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    return data_size_tr


if __name__ == '__main__':
    # file_name = "test"
    file_name = "rawData/train"
    data, label = getSequenceData(file_name)
    print(len(data))
    print(len(label))
    data_e, label_e, seq_length = PadEncode(data, label)
    # writeFasta(file_name, data_e, label_e, seq_length)

    csv_file_path = 'train_label.csv'

    import csv

    # 将数据写入CSV文件
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(label_e)

    print(len(data_e))
    print(len(label_e))
    print(len(seq_length))

    # print(data_e[0])
    # print(label_e[0])
    # print(seq_length[0])

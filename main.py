import os
import time
import random
import argparse
import numpy as np
from Dataset import Dataset
from run_networks import Train


def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ckpt", type=str,
                        help="The input data directory")
    parser.add_argument("--embedding_size", default=1047, type=int,
                        help="The embedding size of CNN")
    parser.add_argument("--linear_size", default=2304, type=int,
                        help="The size of Linear, ours is 2304")
    parser.add_argument("--HGNN_size", default=2304, type=int,
                        help="The output size of HGNN")
    parser.add_argument("--label_size", default=21, type=int,
                        help="The number of label")
    parser.add_argument('--seed', type=int, default=19740322,
                        help="random seed for initialization, 20240406, 19740322")
    parser.add_argument('--max_len', type=int, default=50,
                        help="max length of sequence")
    parser.add_argument('--batch_size', type=int, default=192,
                        help="batch size of model")
    parser.add_argument("--learning_rate", default=0.0018, type=float,
                        help="learning rate of model")
    parser.add_argument('--epochs', type=int, default=200,
                        help="epoch of model")
    parser.add_argument('--length', type=int, default=7872,
                        help="size of train_loader")
    args = parser.parse_args()

    args.device = torch.device('cuda:0')
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    os.makedirs(args.data_dir, exist_ok=True)
    args.check_pt_model_path = os.path.join(args.data_dir, "adaptive_%s.pth" % args.timemark)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    return args


def main(args, filenames, amino_acids):
    print("Loading............")
    dataset = Dataset(train_data_path=f"data/MFTP/Seqvec/train",
                      test_data_path=f"data/MFTP/Seqvec/test",
                      filenames=filenames,
                      amino_acids=amino_acids,
                      args=args)
    print(dataset.label_encode)

    print("Training............")
    training_model = Train(args, dataset, test=False)
    training_model.train()

    print("Testing............")
    training_model = Train(args, dataset, test=True)
    training_model.load_model(args.check_pt_model_path)
    result, y_pred, y_test, feature = training_model.eval(phase='eval')

    return result


if __name__ == '__main__':
    import torch
    torch.autograd.set_detect_anomaly(True)

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
    args = setArgs()
    result = main(args, filenames, amino_acids)

    print("Precision: ", result['aiming'])
    print("Coverage: ", result['coverage'])
    print("Accuracy: ", result['accuracy'])
    print("Absolute True: ", result['absolute_true'])
    print("Absolute False: ", result['absolute_false'])
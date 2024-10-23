import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from scipy.sparse import csr_matrix
from functools import partial
from typing import Union, Optional, List, Iterable, Hashable


def scores(y_test, y_pred, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    SN = tp * 1.0 / ((tp + fn) * 1.0)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return Recall, SN, SP, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp


def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            score_k += 1
    return score_k / n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        sorce_k += (union - intersection) / m
    return sorce_k / n


def evaluate(y_hat, y):
    score_label = y_hat
    aiming_list = []
    coverage_list = []
    accuracy_list = []
    absolute_true_list = []
    absolute_false_list = []
    # print("y_hat: ", y_hat[0])
    # getting prediction label

    # for i, out in enumerate(y_hat[0]):
    #     print(f"y_pred: {out: .5f}")
    # print("y: ", y[0])
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:  # throld
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1
    # print("score_label: ", score_label[0])
    # print("y: ", y[0])
    y_hat = score_label
    aiming = Aiming(y_hat, y)
    aiming_list.append(aiming)
    coverage = Coverage(y_hat, y)
    coverage_list.append(coverage)
    accuracy = Accuracy(y_hat, y)
    accuracy_list.append(accuracy)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_true_list.append(absolute_true)
    absolute_false = AbsoluteFalse(y_hat, y)
    absolute_false_list.append(absolute_false)
    return dict(aiming=aiming, coverage=coverage, accuracy=accuracy, absolute_true=absolute_true,
                absolute_false=absolute_false)


TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]


def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
    if classes is not None:
        mlb = MultiLabelBinarizer(classes, sparse_output=True)
    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
            mlb.fit(None)
        else:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(targets)
    return mlb


def get_psp(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
            classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den


def get_inv_propensity(train_y, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)


def get_threshold(class_freq):
    total_samples = np.sum(class_freq)
    thresholds = (class_freq / (total_samples - class_freq))

    for i, threshold in enumerate(thresholds):
        if(threshold >= 0.1):
            thresholds[i] = 0.5
        else:
            thresholds[i] = 0.5

    return thresholds


if __name__ == '__main__':
    class_freq = np.array([115, 1735, 850, 98, 400, 48, 1079, 82, 732, 1624, 147, 218, 182, 568, 92, 273, 366, 250, 171, 89, 531])
    thresholds = get_threshold(class_freq)

    print(thresholds)

    # probabilities = np.random.random((3, 21))
    # for i in range(len(probabilities)):
    #     for j in range(len(probabilities[i])):
    #         probabilities[i][j] = round(probabilities[i][j], 2)
    #
    # print(probabilities)
    #
    # score_label = probabilities
    # for i in range(len(probabilities)):
    #     for j in range(len(probabilities[i])):
    #         if score_label[i][j] < thresholds[j]:  # throld
    #             score_label[i][j] = 0
    #         else:
    #             score_label[i][j] = 1
    #
    # print(score_label)
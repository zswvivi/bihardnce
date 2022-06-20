# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split as train_test_split


def cal_set_micro_precision(true_sets, pred_sets):
    TP = sum([len(pred_set & true_set) for pred_set, true_set in zip(pred_sets, true_sets)])
    precision = TP / sum([len(pred_set) for pred_set in pred_sets])
    return precision


def cal_set_micro_recall(true_sets, pred_sets):
    TP = sum([len(pred_set & true_set) for pred_set, true_set in zip(pred_sets, true_sets)])
    recall = TP / sum([len(true_set) for true_set in true_sets])
    return recall


def cal_set_micro_f1(true_sets, pred_sets):
    TP = sum([len(pred_set & true_set) for pred_set, true_set in zip(pred_sets, true_sets)])
    precision = TP / sum([len(pred_set) for pred_set in pred_sets])
    recall = TP / sum([len(true_set) for true_set in true_sets])
    f1 = 2 * precision * recall / (precision + recall)
    return {'PRECISION': precision, 'RECALL': recall, 'F1': f1}

def cal_ndcg(true_sets, pred_sets, topk = 5):
    return ndcg_score(true_sets, pred_sets, k = topk)

if __name__ == '__main__':
    pass

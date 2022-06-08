import os
import argparse
import torch
from networks.ganf.GANF import GANF
import numpy as np
from sklearn.metrics import roc_auc_score


def predict_prob(model, test_iterator, evaluate_dir, window_labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(evaluate_dir + "/GANF_SMD_best.pt"))
    A = torch.load(evaluate_dir + "/graph_best.pt").to(device)
    model.eval()

    loss_test = []
    with torch.no_grad():
        for x in test_iterator:
            x = x.unsqueeze(-1).transpose(1, 2)
            x = x.to(device)
            loss = -model.test(x, A.data).cpu().numpy()
            loss_test.append(loss)
    loss_test = np.concatenate(loss_test)
    anomaly_score = loss_test
    anomaly_label = window_labels[-len(anomaly_score) :]

    return anomaly_score, anomaly_label

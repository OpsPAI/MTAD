import os
import sys

import json
import glob
import hashlib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from common.evaluation.spot import SPOT
from common.utils import pprint


def evaluate_all(
    anomaly_score,
    anomaly_label,
    train_anomaly_score=None,
    q=1e-3,
    level=None,
    verbose=True,
):
    # normalize anomaly_score
    print("Normalizing anomaly scores.")
    anomaly_score = normalize_1d(anomaly_score)
    if train_anomaly_score is not None:
        train_anomaly_score = normalize_1d(train_anomaly_score)

    metrics = {}
    # compute auc
    try:
        auc = roc_auc_score(anomaly_label, anomaly_score)
    except ValueError:
        auc = 0
        print("All zero in anomaly label, set auc=0")
    metrics["1.AUC"] = auc

    # compute salience
    # salience = compute_salience(anomaly_score, anomaly_label)
    # metrics["2.Salience"] = salience

    # iterate thresholds
    # _, theta_iter, _, pred = iter_thresholds(
    #     anomaly_score, anomaly_label, metric="f1", normalized=True
    # )
    # _, adjust_pred = point_adjustment(pred, anomaly_label)
    # metrics_iter = compute_point2point(pred, adjust_pred, anomaly_label)
    # metrics_iter["delay"] = compute_delay(anomaly_label, pred)
    # metrics_iter["theta"] = theta_iter
    # metrics["3.Iteration Based"] = metrics_iter

    # # EVT needs anomaly scores on training data for initialization
    # if train_anomaly_score is not None:
    #     print("Finding thresholds via EVT.")
    #     theta_evt, pred_evt = compute_th_evt(
    #         train_anomaly_score, anomaly_score, anomaly_label, q, level
    #     )
    #     _, adjust_pred_evt = point_adjustment(pred_evt, anomaly_label)
    #     metrics_evt = compute_point2point(pred_evt, adjust_pred_evt, anomaly_label)
    #     metrics_evt["delay"] = compute_delay(anomaly_label, pred_evt)
    #     metrics_evt["theta"] = theta_evt
    #     metrics["4.EVT Based"] = metrics_evt

    if verbose:
        print("\n" + "-" * 20 + "\n")
        pprint(metrics)

    return metrics


def normalize_1d(arr):
    est = MinMaxScaler()
    return est.fit_transform(arr.reshape(-1, 1)).reshape(-1)


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def store_benchmarking_results(
    hash_id,
    benchmark_dir,
    dataset,
    entity,
    args,
    model_name,
    anomaly_score,
    anomaly_label,
    time_tracker,
):
    value_store_dir = os.path.join(benchmark_dir, model_name, hash_id, dataset, entity)
    os.makedirs(value_store_dir, exist_ok=True)
    np.savez(os.path.join(value_store_dir, "anomaly_score"), anomaly_score)
    np.savez(os.path.join(value_store_dir, "anomaly_label"), anomaly_label)

    json_pretty_dump(time_tracker, os.path.join(value_store_dir, "time.json"))

    param_store_dir = os.path.join(benchmark_dir, model_name, hash_id)

    param_store = {"cmd": "python {}".format(" ".join(sys.argv))}
    param_store.update(args)

    json_pretty_dump(param_store, os.path.join(param_store_dir, "params.json"))
    print("Store output of {} to {} done.".format(model_name, param_store_dir))
    return os.path.join(benchmark_dir, model_name, hash_id, dataset)


def evaluate_benchmarking_folder(
    folder, benchmarking_dir, hash_id, dataset, model_name
):
    total_adj_f1 = []
    total_train_time = []
    total_test_time = []
    folder_count = 0
    for folder in glob.glob(os.path.join(folder, "*")):
        folder_name = os.path.basename(folder)
        print("Evaluating {}".format(folder_name))

        anomaly_score = np.load(
            os.path.join(folder, "anomaly_score.npz"), allow_pickle=True
        )["arr_0"].item()["test"]

        anomaly_score_train = np.load(
            os.path.join(folder, "anomaly_score.npz"), allow_pickle=True
        )["arr_0"].item()["train"]

        anomaly_label = np.load(os.path.join(folder, "anomaly_label.npz"))[
            "arr_0"
        ].astype(int)
        with open(os.path.join(folder, "time.json")) as fr:
            time = json.load(fr)

        best_f1, best_theta, best_adjust_pred, best_raw_pred = iter_thresholds(
            anomaly_score, anomaly_label, metric="f1", adjustment=True
        )

        try:
            auc = roc_auc_score(anomaly_label, anomaly_score)
        except ValueError as e:
            auc = 0
            print("All zero in anomaly label, set auc=0")

        metrics = {}
        metrics_iter, metrics_evt, theta_iter, theta_evt = evaluate_all(
            anomaly_score, anomaly_label, anomaly_score_train
        )

        total_adj_f1.append(metrics_iter["adj_f1"])
        total_train_time.append(time["train"])
        total_test_time.append(time["test"])

        metrics["metrics_iteration"] = metrics_iter
        metrics["metrics_iteration"]["theta"] = theta_iter
        metrics["metrics_evt"] = metrics_evt
        metrics["metrics_evt"]["theta"] = theta_evt
        # metrics["train_time"] = time["train"]
        # metrics["test_time"] = time["test"]

        print(metrics)
        json_pretty_dump(metrics, os.path.join(folder, "metrics.json"))
        folder_count += 1

    total_adj_f1 = np.array(total_adj_f1)
    adj_f1_mean = total_adj_f1.mean()
    adj_f1_std = total_adj_f1.std()

    train_time_sum = sum(total_train_time)
    test_time_sum = sum(total_test_time)

    with open(
        os.path.join(benchmarking_dir, f"{dataset}_{model_name}.txt"), "a+"
    ) as fw:
        params = " ".join(sys.argv)
        info = f"{hash_id}\tcount:{folder_count}\t{params}\ttrain:{train_time_sum:.4f} test:{test_time_sum:.4f}\tadj f1: [{adj_f1_mean:.4f}({adj_f1_std:.4f})]\n"
        fw.write(info)
    print(info)


if __name__ == "__main__":
    anomaly_label = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    anomaly_score = np.random.uniform(0, 1, size=len(anomaly_label))
    evaluate_all(anomaly_score, anomaly_label)

import imp
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, namedtuple
from common.exp import json_pretty_dump
from ..utils import load_hdf5, load_json, print_to_json
from .metrics import compute_binary_metrics, compute_delay
from .point_adjustment import adjust_pred
from .thresholding import best_th, eps_th, pot_th


def get_comb_key(thresholding, point_adjustment):
    return "{}{}".format(thresholding, "_adjusted" if point_adjustment else "")


def results2csv(results, filepath):
    columns = [
        "uptime",
        "dataset_id",
        "strategy",
        "exp_id",
        "model_id",
        "length",
        "f1_adjusted",
        "pc_adjusted",
        "rc_adjusted",
        "f1",
        "pc",
        "rc",
        "delay",
        "train_time",
        "test_time",
        "nb_epoch",
        "nb_eval_entity",
        "nb_total_entity",
    ]

    filedir = os.path.dirname(filepath)
    os.makedirs(filedir, exist_ok=True)

    total_rows = []
    basic_info = {
        key: value for key, value in results.items() if not isinstance(value, dict)
    }

    for key, value in results.items():
        if isinstance(value, dict):
            row = {"strategy": key, **value, **basic_info}
            total_rows.append(row)

    if os.path.isfile(filepath):
        logging.info(f"File {filepath} exists, loading directly.")
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame()
    total_rows.extend(df.to_dict(orient="records"))
    pd.DataFrame(total_rows, columns=columns).to_csv(filepath, index=False)
    logging.info(f"Appended exp results to {filepath}.")


class Evaluator:
    """

    th (str): options: "best", "eps", "pot"
    """

    def __init__(
        self,
        metrics,
        thresholding="best",
        pot_params={"q": 1e-3, "level": 0.99, "dynamic": False},
        best_params={"target_metric": "f1", "target_direction": "max"},
        point_adjustment=False,
        reverse_score=False
    ):
        if isinstance(thresholding, str):
            thresholding = [thresholding]
        if isinstance(point_adjustment, str):
            point_adjustment = [point_adjustment]

        self.thresholding = thresholding
        self.metrics = metrics
        self.best_params = best_params
        self.pot_params = pot_params
        self.point_adjustment = point_adjustment
        self.reverse_score = reverse_score

    def score2pred(
        self,
        thresholding,
        anomaly_score,
        anomaly_label,
        train_anomaly_score=None,
        point_adjustment=False,
    ):
        if self.reverse_score:
            anomaly_score = -anomaly_score


        pred_results = {"anomaly_pred": None, "anomaly_pred_adjusted": None, "th": None}

        if thresholding == "best":
            th = best_th(
                anomaly_score,
                anomaly_label,
                point_adjustment=point_adjustment,
                **self.best_params,
            )
        if thresholding == "pot":
            th = pot_th(train_anomaly_score, anomaly_score, **self.pot_params)
        if thresholding == "eps":
            th = eps_th(train_anomaly_score, reg_level=1)
            

        anomaly_pred = (anomaly_score >= th).astype(int)

        pred_results["anomaly_pred"] = anomaly_pred
        pred_results["th"] = th
        if self.point_adjustment:
            pred_results["anomaly_pred_adjusted"] = adjust_pred(
                anomaly_pred, anomaly_label
            )
        return pred_results

    def eval(
        self,
        anomaly_label,
        anomaly_score=None,
        train_anomaly_score=None,
    ):
        eval_results = {}
        for point_adjustment in self.point_adjustment:
            for thresholding in self.thresholding:
                eval_results_tmp = {}

                pred_results = self.score2pred(
                    thresholding,
                    anomaly_score,
                    anomaly_label,
                    train_anomaly_score,
                    point_adjustment,
                )
                eval_results_tmp["th"] = pred_results["th"]
                anomaly_pred = pred_results["anomaly_pred"]

                eval_results_tmp.update(
                    self.cal_metrics(anomaly_pred, anomaly_label, point_adjustment)
                )

                key = get_comb_key(thresholding, point_adjustment)
                eval_results[key] = eval_results_tmp
        return eval_results

    def cal_metrics(self, anomaly_pred, anomaly_label, point_adjustment):
        logging.info(
            "Pred pos {}/{}, Label pos {}/{}".format(
                anomaly_pred.sum(),
                anomaly_pred.shape[0],
                anomaly_label.sum(),
                anomaly_label.shape[0],
            )
        )
        eval_metrics = {"length": anomaly_pred.shape[0]}
        for metric in self.metrics:
            if metric in ["f1", "pc", "rc"]:
                eval_metrics.update(
                    compute_binary_metrics(
                        anomaly_pred,
                        anomaly_label,
                        point_adjustment,
                    )
                )
            if metric == "delay":
                eval_metrics["delay"] = compute_delay(anomaly_pred, anomaly_label)
        return eval_metrics

    def eval_exp(
        self, exp_folder, entities, merge_folder, extra_params, eval_single=False
    ):
        eval_results = {
            "dataset_id": extra_params["dataset_id"],
            "exp_id": extra_params["exp_id"],
            "model_id": extra_params["model_id"],
            "train_time": 0,
            "test_time": 0,
            "nb_epoch": 0,
            "nb_eval_entity": 0,
            "nb_total_entity": len(entities),
            "uptime": extra_params["uptime"],
        }
        # score to pred, for every entity
        merge_dict = {
            "anomaly_pred": defaultdict(list),
            "anomaly_label": defaultdict(list),
        }
        for entity in entities:
            entity_folder = os.path.join(exp_folder, entity)
            try:
                score_dict = load_hdf5(
                    os.path.join(entity_folder, f"score_{entity}.hdf5")
                )
                time_track = load_json(os.path.join(entity_folder, "time.json"))
            except:
                logging.warn("Failed to load entity {}.".format(entity))
                continue

            # merge efficiency info to eval_results
            eval_results["train_time"] += float(time_track["train_time"])
            eval_results["test_time"] += float(time_track["test_time"])
            eval_results["nb_epoch"] += int(time_track["nb_epoch"])
            eval_results["nb_eval_entity"] += 1

            thresholds = {}
            eval_results_single = defaultdict(dict)
            for point_adjustment in self.point_adjustment:
                for thresholding in self.thresholding:
                    EvalKey = namedtuple("key", ["point_adjustment", "thresholding"])
                    eval_key = EvalKey(point_adjustment, thresholding)

                    pred_results = self.score2pred(
                        thresholding,
                        score_dict["anomaly_score"],
                        score_dict["anomaly_label"],
                        score_dict["train_anomaly_score"],
                        point_adjustment,
                    )

                    key = get_comb_key(eval_key.thresholding, eval_key.point_adjustment)
                    thresholds[key] = pred_results["th"]
                    merge_dict["anomaly_pred"][eval_key].append(
                        pred_results["anomaly_pred"]
                    )
                    merge_dict["anomaly_label"][eval_key].append(
                        score_dict["anomaly_label"]
                    )

                    if eval_single:
                        eval_results_single[key]["th"] = pred_results["th"]
                        eval_results_single[key].update(
                            self.cal_metrics(
                                pred_results["anomaly_pred"],
                                score_dict["anomaly_label"],
                                point_adjustment,
                            )
                        )
            if eval_single:
                json_pretty_dump(
                    eval_results_single,
                    os.path.join(entity_folder, "eval_results.json"),
                )

            json_pretty_dump(thresholds, os.path.join(entity_folder, "thresholds.json"))

        # merge effectiveness info to eval_results
        for eval_key in merge_dict["anomaly_pred"].keys():
            key = get_comb_key(eval_key.thresholding, eval_key.point_adjustment)

            pred_cat = np.concatenate(merge_dict["anomaly_pred"][eval_key])
            label_cat = np.concatenate(merge_dict["anomaly_label"][eval_key])
            eval_result_tmp = self.cal_metrics(
                pred_cat, label_cat, eval_key.point_adjustment
            )
            eval_result_tmp["length"] = pred_cat.shape[0]
            eval_results[key] = eval_result_tmp

        logging.info(print_to_json(eval_results))
        logging.info(
            "Evaluated {}/{} entities.".format(
                eval_results["nb_eval_entity"], len(entities)
            )
        )

        results2csv(
            eval_results,
            os.path.join(merge_folder, extra_params["dataset_id"], "bench_results.csv"),
        )

    def eval_exp_simple(
        self, exp_folder, entities, merge_folder, extra_params, eval_single=False
    ):
        merge_dict = {
            "anomaly_pred": defaultdict(list),
            "anomaly_label": defaultdict(list),
        }
        for entity in entities:
            entity_folder = os.path.join(exp_folder, entity)
            try:
                score_dict = load_hdf5(
                    os.path.join(entity_folder, f"score_{entity}.hdf5")
                )
                time_track = load_json(os.path.join(entity_folder, "time.json"))
            except:
                logging.warn("Failed to load entity {}.".format(entity))
                continue

            for point_adjustment in self.point_adjustment:
                for thresholding in self.thresholding:
                    EvalKey = namedtuple("key", ["point_adjustment", "thresholding"])
                    eval_key = EvalKey(point_adjustment, thresholding)

                    pred_results = self.score2pred(
                        thresholding,
                        score_dict["anomaly_score"],
                        score_dict["anomaly_label"],
                        score_dict["train_anomaly_score"],
                        point_adjustment,
                    )
                    merge_dict["anomaly_pred"][eval_key].append(
                        pred_results["anomaly_pred"]
                    )
                    merge_dict["anomaly_label"][eval_key].append(
                        score_dict["anomaly_label"]
                    )

        eval_results = {}
        # merge effectiveness info to eval_results
        for eval_key in merge_dict["anomaly_pred"].keys():
            key = get_comb_key(eval_key.thresholding, eval_key.point_adjustment)

            pred_cat = np.concatenate(merge_dict["anomaly_pred"][eval_key])
            label_cat = np.concatenate(merge_dict["anomaly_label"][eval_key])
            eval_result_tmp = self.cal_metrics(
                pred_cat, label_cat, eval_key.point_adjustment
            )
            eval_result_tmp["length"] = pred_cat.shape[0]
            eval_results[key] = eval_result_tmp

            logging.info("{} pred_sum: {}".format(eval_key, pred_cat.sum()))

        logging.info(print_to_json(eval_results))

from cgi import print_form
import logging
import numpy as np
import more_itertools as mit
from .metrics import compute_binary_metrics
from .spot import SPOT


def pot_th(train_anomaly_score, anomaly_score, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    logging.info(f"Computing the threshold using POT with q={q}, level={level}...")
    logging.info(
        "[POT] Train score max: {}, min: {}".format(
            train_anomaly_score.max(), train_anomaly_score.min()
        )
    )
    logging.info(
        "[POT] Test score max: {}, min: {}".format(
            anomaly_score.max(), anomaly_score.min()
        )
    )
    print(train_anomaly_score.shape, anomaly_score.shape)

    pot_th = None
    if not isinstance(level, list):
        level = [level]
    for l in level:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(train_anomaly_score, anomaly_score)
            s.initialize(level=l, min_extrema=False)  # Calibration step
            ret = s.run(dynamic=dynamic, with_alarm=False)
            pot_th = np.mean(ret["thresholds"])
            logging.info(f"Hit level={l}")
            break
        except:
            pass
    if pot_th is None:
        pot_th = np.percentile(anomaly_score, level[0] * 100)
        logging.info(
            "POT cannot find the threshold, use {}% percentile {}".format(
                level[0] * 100, pot_th
            )
        )
    return pot_th


def eps_th(train_anomaly_score, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    logging.info("Computing the threshold with eps...")
    e_s = train_anomaly_score
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(
            -1,
        )
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def best_th(
    anomaly_score,
    anomaly_label,
    target_metric="f1",
    target_direction="max",
    point_adjustment=False,
):
    logging.info("Searching for the best threshod..")
    search_range = np.linspace(0, 1, 100)
    search_history = []
    if point_adjustment:
        target_metric = target_metric + "_adjusted"

    for anomaly_percent in search_range:
        theta = np.percentile(anomaly_score, 100 * (1 - anomaly_percent))
        pred = (anomaly_score >= theta).astype(int)

        metric_dict = compute_binary_metrics(pred, anomaly_label, point_adjustment)
        current_value = metric_dict[target_metric]

        logging.debug(f"th={theta}, {target_metric}={current_value}")

        search_history.append(
            {
                "best_value": current_value,
                "best_theta": theta,
                "target_metric": target_metric,
                "target_direction": target_direction,
            }
        )

    result = (
        max(search_history, key=lambda x: x["best_value"])
        if target_direction == "max"
        else min(search_history, key=lambda x: x["best_value"])
    )
    return result["best_theta"]

import logging
import yaml
import json
import os
from .utils import save_hdf5


BENCHMARK_DIR = "./benchmark_results"


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            {str(k): str(v) for k, v in obj.items()},
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def store_entity(
    params,
    entity,
    train_anomaly_score,
    anomaly_score,
    anomaly_label,
    eval_results={},
    time_tracker={},
):
    exp_folder = params["model_root"]
    entity_folder = os.path.join(exp_folder, entity)
    os.makedirs(entity_folder, exist_ok=True)

    # save params
    with open(os.path.join(exp_folder, "params.yaml"), "w") as fw:
        yaml.dump(params, fw)

    # save results
    json_pretty_dump(eval_results, os.path.join(entity_folder, "eval_results.json"))

    # save time
    json_pretty_dump(time_tracker, os.path.join(entity_folder, "time.json"))

    # save scores
    score_dict = {
        "anomaly_label": anomaly_label,
        "anomaly_score": anomaly_score,
        "train_anomaly_score": train_anomaly_score,
    }
    save_hdf5(os.path.join(entity_folder, f"score_{entity}.hdf5"), score_dict)

    logging.info(f"Saving results for {entity} done.")


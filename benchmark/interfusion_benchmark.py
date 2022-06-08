# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")

import logging
from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from networks.InterFusion import InterFusion
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity

seed_everything()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./benchmark_config/",
        help="The config directory.",
    )
    parser.add_argument("--expid", type=str, default="interfusion_SMD")
    parser.add_argument("--gpu", type=int, default=-1)
    args = vars(parser.parse_args())

    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        nrows=params["nrows"],
    )
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))
        train = data_dict[entity]["train"]
        valid = data_dict[entity].get("valid", None)
        test, test_label = (
            data_dict[entity]["test"],
            data_dict[entity]["test_label"],
        )

        model = InterFusion(
            dataset=params["dataset"],
            model_root=params["model_root"],
            dim=params["dim"],
        )
        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(
            x_train=train,
            x_valid=valid,
            lr=params["lr"],
            window_size=params["window_size"],
            batch_size=params["batch_size"],
            pretrain_max_epoch=params["pretrain_max_epoch"],
            max_epoch=params["nb_epoch"],
        )
        tt.train_end()

        train_anomaly_score = model.predict_prob(train, None)

        tt.test_start()
        anomaly_score, anomaly_label = model.predict_prob(test, test_label)
        tt.test_end()

        store_entity(
            params,
            entity,
            train_anomaly_score,
            anomaly_score,
            anomaly_label,
            time_tracker=tt.get_data(),
        )
        del model
    evaluator.eval_exp(
        exp_folder=params["model_root"],
        entities=params["entities"],
        merge_folder=params["benchmark_dir"],
        extra_params=params,
    )

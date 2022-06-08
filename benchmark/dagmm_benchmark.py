import sys

sys.path.append("../")
import logging
import argparse
from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from networks.dagmm.dagmm import DAGMM

seed_everything()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./benchmark_config/",
        help="The config directory.",
    )
    parser.add_argument("--expid", type=str, default="dagmm_SMD")
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

    # preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))

        train = data_dict[entity]["train"]
        test = data_dict[entity]["test"]
        test_label = data_dict[entity]["test_label"]

        model = DAGMM(
            comp_hiddens=params["compression_hiddens"],
            est_hiddens=params["estimation_hiddens"],
            est_dropout_ratio=params["estimation_dropout_ratio"],
            minibatch_size=params["batch_size"],
            epoch_size=params["nb_epoch"],
            learning_rate=params["lr"],
            lambda1=params["lambdaone"],
            lambda2=params["lambdatwo"],
        )

        # predict anomaly score
        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(train)
        tt.train_end()

        train_anomaly_score = model.predict_prob(test)
        tt.test_start()
        anomaly_score = model.predict_prob(test)
        tt.test_end()

        anomaly_label = test_label

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

import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append("../")
import logging
from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from networks.RANS import RANSynCoders


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
    parser.add_argument("--expid", type=str, default="rans_SMD")
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
        valid_ratio=params["valid_ratio"],
        dim=params["dim"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        nrows=params["nrows"],
    )

    # preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))
        x_train = data_dict[entity]["train"]
        x_test = data_dict[entity]["test"]

        N = 5 * round((x_train.shape[1] / 3) / 5)
        z = int((N / 2) - 1)

        model = RANSynCoders(
            n_estimators=N,
            max_features=N,
            encoding_depth=params["encoder_layers"],
            latent_dim=z,
            decoding_depth=params["decoder_layers"],
            activation=params["activation"],
            output_activation=params["output_activation"],
            delta=params["delta"],
            synchronize=params["synchronize"],
            max_freqs=params["S"],
        )

        tt = TimeTracker(nb_epoch=params["nb_epoch"])
        tt.train_start()

        model.fit(
            x_train,
            epochs=params["nb_epoch"],
            batch_size=params["batch_size"],
            freq_warmup=params["freq_warmup"],
            sin_warmup=params["sin_warmup"],
        )
        tt.train_end()

        train_anomaly_score = model.predict_prob(
            x_train, N, batch_size=10 * params["batch_size"]
        )

        tt.test_start()
        anomaly_score = model.predict_prob(
            x_test, N, batch_size=10 * params["batch_size"]
        )
        tt.test_end()

        anomaly_label = data_dict[entity]["test_label"]

        store_entity(
            params,
            entity,
            train_anomaly_score,
            anomaly_score,
            anomaly_label,
            time_tracker=tt.get_data(),
        )
    evaluator.eval_exp(
        exp_folder=params["model_root"],
        entities=params["entities"],
        merge_folder=params["benchmark_dir"],
        extra_params=params,
    )

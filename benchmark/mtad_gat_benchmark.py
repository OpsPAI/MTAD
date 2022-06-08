import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append("../")
import logging
from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from networks.mtad_gat import MTAD_GAT


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
    parser.add_argument("--expid", type=str, default="mtad_gat_SMD")
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

    # sliding windows
    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))
        windows = window_dict[entity]
        train_windows = windows["train_windows"]
        test_windows = windows["test_windows"]

        train_loader, _, test_loader = get_dataloaders(
            train_windows,
            test_windows,
            next_steps=1,
            batch_size=params["batch_size"],
            shuffle=params["shuffle"],
            num_workers=params["num_workers"],
        )

        model = MTAD_GAT(
            n_features=params["dim"],
            window_size=params["window_size"],
            out_dim=params["dim"],
            kernel_size=params["kernel_size"],
            feat_gat_embed_dim=params["feat_gat_embed_dim"],
            time_gat_embed_dim=params["time_gat_embed_dim"],
            use_gatv2=params["use_gatv2"],
            gru_n_layers=params["gru_n_layers"],
            gru_hid_dim=params["gru_hid_dim"],
            forecast_n_layers=params["forecast_n_layers"],
            forecast_hid_dim=params["forecast_hid_dim"],
            recon_n_layers=params["recon_n_layers"],
            recon_hid_dim=params["recon_hid_dim"],
            dropout=params["dropout"],
            alpha=params["alpha"],
            device=params["device"],
        )

        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(
            train_loader,
            val_loader=None,
            n_epochs=params["nb_epoch"],
            batch_size=params["batch_size"],
            init_lr=params["init_lr"],
            model_root=params["model_root"],
        )
        tt.train_end()

        train_anomaly_score = model.predict_prob(train_loader, gamma=params["gamma"])

        tt.test_start()
        anomaly_score, anomaly_label = model.predict_prob(
            test_loader, gamma=params["gamma"], window_labels=windows["test_label"]
        )
        tt.test_end()

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

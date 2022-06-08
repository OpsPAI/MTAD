import logging
import os
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

data_path_dict = {
    "SMD": "./datasets/anomaly/SMD/processed",
    "SMAP": "./datasets/anomaly/SMAP-MSL/processed_SMAP",
    "MSL": "./datasets/anomaly/SMAP-MSL/processed_MSL",
    "WADI": "./datasets/anomaly/WADI/processed",
    "SWAT": "./datasets/anomaly/SWAT/processed",
    "WADI_SPLIT": "./datasets/anomaly/WADI_SPLIT/processed",
    "SWAT_SPLIT": "./datasets/anomaly/SWAT_SPLIT/processed",
}


def get_data_dim(dataset):
    if "SMAP" in dataset:
        return 25
    elif "MSL" in dataset:
        return 55
    elif "SMD" in dataset:
        return 38
    elif "WADI" in dataset:
        return 93
    elif "SWAT" in dataset:
        return 40
    else:
        raise ValueError("unknown dataset " + str(dataset))


def load_dataset(
    data_root,
    entities,
    valid_ratio,
    dim,
    test_label_postfix,
    test_postfix,
    train_postfix,
    nan_value=0,
    nrows=None,
):
    """
    use_dim: dimension used in multivariate timeseries
    """
    logging.info("Loading data from {}".format(data_root))

    data = defaultdict(dict)
    total_train_len, total_valid_len, total_test_len = 0, 0, 0
    for dataname in entities:
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, train_postfix)), "rb"
        ) as f:
            train = pickle.load(f).reshape((-1, dim))[0:nrows, :]
            if valid_ratio > 0:
                split_idx = int(len(train) * valid_ratio)
                train, valid = train[:-split_idx], train[-split_idx:]
                data[dataname]["valid"] = np.nan_to_num(valid, nan_value)
                total_valid_len += len(valid)
            data[dataname]["train"] = np.nan_to_num(train, nan_value)
            total_train_len += len(train)
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, test_postfix)), "rb"
        ) as f:
            test = pickle.load(f).reshape((-1, dim))[0:nrows, :]
            data[dataname]["test"] = np.nan_to_num(test, nan_value)
            total_test_len += len(test)
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, test_label_postfix)), "rb"
        ) as f:
            data[dataname]["test_label"] = pickle.load(f).reshape(-1)[0:nrows]
    logging.info("Loading {} entities done.".format(len(entities)))
    logging.info(
        "Train/Valid/Test: {}/{}/{} lines.".format(
            total_train_len, total_valid_len, total_test_len
        )
    )

    return data


class sliding_window_dataset(Dataset):
    def __init__(self, data, next_steps=0):
        self.data = data
        self.next_steps = next_steps

    def __getitem__(self, index):
        if self.next_steps == 0:
            x = self.data[index]
            return x
        else:
            x = self.data[index, 0 : -self.next_steps]
            y = self.data[index, -self.next_steps :]
            return x, y

    def __len__(self):
        return len(self.data)


def get_dataloaders(
    train_data,
    test_data,
    valid_data=None,
    next_steps=0,
    batch_size=32,
    shuffle=True,
    num_workers=1,
):

    train_loader = DataLoader(
        sliding_window_dataset(train_data, next_steps),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        sliding_window_dataset(test_data, next_steps),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if valid_data is not None:
        valid_loader = DataLoader(
            sliding_window_dataset(valid_data, next_steps),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        valid_loader = None
    return train_loader, valid_loader, test_loader

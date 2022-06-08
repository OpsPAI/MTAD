from .train import ExpConfig, TrainConfig, fit
from .predict import predict_prob


class InterFusion:
    def __init__(self, dataset, model_root, dim):
        self.dataset = dataset
        self.model_root = model_root
        self.dim = dim
        self.train_exp = None

    def fit(
        self,
        x_train,
        x_valid,
        lr,
        window_size,
        batch_size,
        pretrain_max_epoch,
        max_epoch,
    ):
        self.train_exp = fit(
            self.dataset,
            self.model_root,
            x_train,
            x_valid,
            self.dim,
            lr,
            window_size,
            batch_size,
            pretrain_max_epoch,
            max_epoch,
        )

    def predict_prob(self, x_test, y_test):
        return predict_prob(x_test, y_test, self.train_exp.config, self.model_root)

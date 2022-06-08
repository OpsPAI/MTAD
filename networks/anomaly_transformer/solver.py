import logging
import torch
import torch.nn as nn
import numpy as np
import os
import time

from common.utils import set_device

from .model.AnomalyTransformer import Anomaly_Transformer

torch.autograd.set_detect_anomaly(True)


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        logging.info("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter = counter + 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            logging.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(),
            os.path.join(path, "checkpoint.pth"),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class AnomalyTransformer(object):
    DEFAULTS = {}

    def __init__(self, **config):

        self.__dict__.update(AnomalyTransformer.DEFAULTS, **config)

        self.device = set_device(self.device)
        self.build_model()
        self.criterion = nn.MSELoss(reduction="mean")

    def build_model(self):
        self.model = Anomaly_Transformer(
            win_size=self.win_size,
            device=self.device,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=3,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss = (
                    series_loss
                    + torch.mean(
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ).detach(),
                        )
                    )
                    + torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ).detach(),
                            series[u],
                        )
                    )
                )
                prior_loss = (
                    prior_loss
                    + torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ),
                            series[u].detach(),
                        )
                    )
                    + torch.mean(
                        my_kl_loss(
                            series[u].detach(),
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ),
                        )
                    )
                )
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def fit(self, train_loader, vali_loader):
        self.train_loader = train_loader
        self.vali_loader = vali_loader

        logging.info("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count = iter_count + 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss = (
                        series_loss
                        + torch.mean(
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                                series[u],
                            )
                        )
                    )
                    prior_loss = (
                        prior_loss
                        + torch.mean(
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                series[u].detach(),
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                            )
                        )
                    )
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    logging.info(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                # self.optimizer.step()

            logging.info(
                "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            )
            train_loss = np.average(loss1_list)

            if self.vali_loader is not None:
                vali_loss1, vali_loss2 = self.vali(self.vali_loader)
                early_stopping(vali_loss1, vali_loss2, self.model, path)
                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break
            else:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path, "checkpoint.pth"),
                )
                vali_loss1 = 0
            logging.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def predict_prob(self, test_loader, windows_label=None):
        self.test_loader = test_loader
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), "checkpoint.pth"))
        )
        self.model.eval()
        temperature = 50

        logging.info("======================TEST MODE======================")
        criterion = nn.MSELoss(reduction="none")
        # (2) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, input_data in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss = series_loss + (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = prior_loss + (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0)
        anomaly_score = np.array(attens_energy).mean(axis=1)

        if windows_label is not None:
            windows_label = (np.sum(windows_label, axis=1) >= 1) + 0
            return anomaly_score, windows_label
        else:
            return anomaly_score

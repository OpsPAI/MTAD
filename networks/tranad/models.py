import logging
import math
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from .dlutils import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from common.utils import set_device


class TranAD(nn.Module):
    def __init__(self, feats, window_size, lr, model_root, device):
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.n_feats = feats
        self.n_window = window_size
        self.device = set_device(device)
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

        self.init_model(lr, model_root)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

    def init_model(self, lr, model_root, retrain=True, test=False):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        if os.path.exists(model_root) and (not retrain or test):
            logging.info("Loading pre-trained model")
            checkpoint = torch.load(os.path.join(model_root, "model.pt"))
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logging.info("Creating new model: TranAD")

        self.optimizer = optimizer
        self.scheduler = scheduler
        logging.info("Finish model initialization.")

    def fit(self, nb_epoch, dataloader, training=True):
        self.to(self.device)
        for epoch in range(1, nb_epoch + 1):
            mse_func = nn.MSELoss(reduction="none")
            n = epoch + 1
            l1s = []
            if training:
                logging.info("Training epoch: {}".format(epoch))
                for d in dataloader:
                    d = d.to(self.device)
                    local_bs = d.shape[0]
                    window = d.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, self.n_feats)
                    z = self(window, elem)
                    l1 = (
                        mse_func(z, elem)
                        if not isinstance(z, tuple)
                        else (1 / n) * mse_func(z[0], elem)
                        + (1 - 1 / n) * mse_func(z[1], elem)
                    )
                    if isinstance(z, tuple):
                        z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                self.scheduler.step()
                logging.info("Epoch: {} finished.".format(epoch))

    def predict_prob(self, test_iterator, label_windows=None):
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        for d in test_iterator:
            d = d.to(self.device)
            bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, self.n_feats)
            z = self(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            loss = mse_func(z, elem)[0]
            loss_steps.append(loss.detach().cpu().numpy())
        anomaly_score = np.concatenate(loss_steps).mean(axis=1)
        if label_windows is None:
            return anomaly_score
        else:
            anomaly_label = (np.sum(label_windows, axis=1) >= 1) + 0
            return anomaly_score, anomaly_label

import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_device

from .modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)


class MTAD_GAT(nn.Module):
    """MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        device="cpu",
    ):
        super(MTAD_GAT, self).__init__()

        window_size = window_size - 1
        self.n_features = n_features
        self.device = set_device(device)
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout
        )
        self.recon_model = ReconstructionModel(
            window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout
        )

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons

    def fit(
        self,
        train_loader,
        val_loader=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        model_root="output/",
        print_every=1,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = nn.MSELoss()
        self.recon_criterion = nn.MSELoss()
        self.print_every = print_every

        logging.info(f"Training model for {self.n_epochs} epochs..")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=init_lr)
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self(x)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            # Evaluate on validation set
            epoch_time = time.time() - epoch_start

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
                s += f" [{epoch_time:.1f}s]"
                logging.info(s)

        if val_loader is None:
            self.save(os.path.join(model_root, "model.pt"))

        train_time = int(time.time() - train_start)
        logging.info(f"-- Training done in {train_time}s.")

    def predict_prob(self, data_loader, gamma, window_labels=None):
        self.gamma = gamma
        self.eval()
        preds = []
        recons = []
        actual = []
        self.to(self.device)
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat, _ = self(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())
                actual.append(
                    x[:, -1, :].detach().cpu().numpy()
                )  # take the last observation as actual

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)

        anomaly_scores = np.zeros_like(actual)
        for i in range(preds.shape[1]):
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
                (recons[:, i] - actual[:, i]) ** 2
            )
            anomaly_scores[:, i] = a_score
        anomaly_scores = np.mean(anomaly_scores, 1)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

    def save(self, file_path):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.load_state_dict(torch.load(file_path, map_location=self.device))

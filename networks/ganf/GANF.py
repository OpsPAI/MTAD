import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.ganf.NF import MAF, RealNVP
from torch.nn.utils import clip_grad_value_
from common.utils import set_device
from torch.nn.init import xavier_uniform_


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum("nkld,kj->njld", h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class GANF(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        dropout=0.1,
        model="MAF",
        batch_norm=True,
        model_root="./checkpoint",
        device="cpu",
    ):
        super(GANF, self).__init__()
        self.device = set_device(device)
        self.model_root = model_root
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            self.nf = MAF(
                n_blocks,
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size=hidden_size,
                batch_norm=batch_norm,
                activation="tanh",
            )
        else:
            self.nf = RealNVP(
                n_blocks,
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size=hidden_size,
                batch_norm=batch_norm,
            )

    def forward(self, x, A):

        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape(
            [full_shape[0], -1]
        )  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def locate(self, x, A):
        # x: N X K X L X D
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape(
            [full_shape[0], full_shape[1], -1]
        )  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=2)

        return log_prob

    def predict_prob(self, test_iterator, window_labels=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        model_file = os.path.join(self.model_root, "model.pt")
        graph_file = os.path.join(self.model_root, "graph.pt")
        self.load_state_dict(torch.load(model_file, map_location=self.device))
        A = torch.load(graph_file).to(device)
        self.eval()

        loss_test = []
        with torch.no_grad():
            for x in test_iterator:
                x = x.unsqueeze(-1).transpose(1, 2)
                x = x.to(device)
                loss = -self.test(x, A.data).cpu().numpy()
                loss_test.append(loss)
        loss_test = np.concatenate(loss_test)
        anomaly_score = loss_test
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_score, anomaly_label
        else:
            return anomaly_score

    def fit(
        self,
        train_iterator,
        valid_iterator=None,
        n_sensor=None,
        weight_decay=5e-4,
        n_epochs=1,
        lr=2e-3,
        h_tol=1e-4,
        rho_max=1e16,
        lambda1=0.0,
        rho_init=1.0,
        alpha_init=0.0,
    ):
        self.to(self.device)

        logging.info("Loading dataset")

        rho = rho_init
        alpha = alpha_init
        lambda1 = lambda1
        h_A_old = np.inf

        # initialize A
        init = torch.zeros([n_sensor, n_sensor])
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        A = torch.tensor(init, requires_grad=True, device=self.device)
        A = A.to(self.device)

        optimizer = torch.optim.Adam(
            [
                {"params": self.parameters(), "weight_decay": weight_decay},
                {"params": [A]},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

        loss_best = 100
        epoch = 0
        for _ in range(n_epochs):
            loss_train = []
            epoch += 1
            self.train()
            for x in train_iterator:
                x = x.unsqueeze(-1).transpose(1, 2)
                x = x.to(self.device)

                optimizer.zero_grad()
                loss = -self(x, A)
                h = torch.trace(torch.matrix_exp(A * A)) - n_sensor
                total_loss = loss + 0.5 * rho * h * h + alpha * h

                total_loss.backward()
                clip_grad_value_(self.parameters(), 1)
                optimizer.step()
                loss_train.append(loss.item())
                A.data.copy_(torch.clamp(A.data, min=0, max=1))
            logging.info(
                "Epoch: {}, train loss: {:.2f}".format(epoch, np.mean(loss_train))
            )
            # eval
            self.eval()
            loss_val = []
            if valid_iterator is not None:
                with torch.no_grad():
                    for x in valid_iterator:
                        x = x.unsqueeze(-1).transpose(1, 2)
                        x = x.to(self.device)
                        loss = -self.test(x, A.data).cpu().numpy()
                        loss_val.append(loss)
                loss_val = np.concatenate(loss_val)
                loss_val = np.nan_to_num(loss_val)
                if np.mean(loss_val) < loss_best:
                    loss_best = np.mean(loss_val)
                    logging.info("save model {} epoch".format(epoch))
                    torch.save(A.data, os.path.join(self.model_root, "graph.pt"))
                    torch.save(
                        self.state_dict(),
                        os.path.join(self.model_root, "model.pt"),
                    )
            else:
                torch.save(A.data, os.path.join(self.model_root, "graph.pt"))
                torch.save(self.state_dict(), os.path.join(self.model_root, "model.pt"))

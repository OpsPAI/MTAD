import logging
import numpy as np
import torch
import torch.nn as nn
from common.utils import set_device
import torch.utils.data as data_utils


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size, device):
        super().__init__()
        self.w_size = w_size
        self.z_size = z_size
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
        self.device = set_device(device)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        return loss1, loss2

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        return {"val_loss1": loss1, "val_loss2": loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x["val_loss1"] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {"val_loss1": epoch_loss1.item(), "val_loss2": epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(
                epoch, result["val_loss1"], result["val_loss2"]
            )
        )

    def fit(
        self, windows_train, windows_val, epochs, batch_size, opt_func=torch.optim.Adam
    ):
        self.to(self.device)
        train_loader = torch.utils.data.DataLoader(
            data_utils.TensorDataset(
                torch.from_numpy(windows_train)
                .float()
                .view(([windows_train.shape[0], self.w_size]))
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        if windows_val is not None:
            val_loader = torch.utils.data.DataLoader(
                data_utils.TensorDataset(
                    torch.from_numpy(windows_val)
                    .float()
                    .view(([windows_val.shape[0], self.w_size]))
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            val_loader = None

        training(
            epochs,
            self,
            train_loader,
            val_loader,
            opt_func=opt_func,
            device=self.device,
        )

    def predict_prob(self, windows_test, batch_size, windows_label=None):
        self.to(self.device)
        test_loader = torch.utils.data.DataLoader(
            data_utils.TensorDataset(
                torch.from_numpy(windows_test)
                .float()
                .view(([windows_test.shape[0], self.w_size]))
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        results = testing(self, test_loader, device=self.device)
        if len(results) >= 2:
            y_pred = np.concatenate(
                [
                    torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                    results[-1].flatten().detach().cpu().numpy(),
                ]
            )
        else:
            y_pred = (results[-1].flatten().detach().cpu().numpy(),)
        if windows_label is not None:
            windows_label = (np.sum(windows_label, axis=1) >= 1) + 0
            return y_pred, windows_label
        else:
            return y_pred


def evaluate(model, val_loader, n, device="cpu"):
    outputs = [
        model.validation_step(to_device(batch, device), n) for [batch] in val_loader
    ]
    return model.validation_epoch_end(outputs)


def training(
    epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam, device="cpu"
):
    history = []
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder1.parameters())
    )
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
    )
    for epoch in range(epochs):
        logging.info(f"Training epoch: {epoch}..")
        for [batch] in train_loader:
            batch = to_device(batch, device)
            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        if val_loader is not None:
            result = evaluate(model, val_loader, epoch + 1, device)
            model.epoch_end(epoch, result)
            history.append(result)
            return history
        logging.info(f"Training epoch: {epoch} done.")


def testing(model, test_loader, alpha=0.5, beta=0.5, device="cpu"):
    with torch.no_grad():
        model.eval()
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(
                alpha * torch.mean((batch - w1) ** 2, axis=1)
                + beta * torch.mean((batch - w2) ** 2, axis=1)
            )
    return results

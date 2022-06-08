## Unit test only start
# import torch
# import sys
import torch
from torch import nn
from networks.lstm.wrappers import TimeSeriesEncoder


class LSTM(TimeSeriesEncoder):
    """
    Encoder of a time series using a LSTM, ccomputing a linear transformation
    of the output of an LSTM

    Takes as input a three-dimensional tensor (`B`, `L`, `C`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a two-dimensional tensor (`B`, `C`).
    """

    def __init__(
        self,
        in_channels,
        hidden_size=64,
        num_layers=1,
        dropout=0,
        prediction_length=1,
        prediction_dims=[],
        **kwargs,
    ):
        super().__init__(architecture="LSTM", **kwargs)

        self.prediction_dims = (
            prediction_dims if prediction_dims else list(range(in_channels))
        )
        self.prediction_length = prediction_length

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        clf_input_dim = hidden_size
        final_output_dim = prediction_length * len(self.prediction_dims)

        self.predcitor = nn.Linear(clf_input_dim, final_output_dim)

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss(reduction="none")

        self.compile()

    def forward(self, batch_window):
        # batch_window = batch_window.permute(0, 2, 1)  # b x win x ts_dim
        self.batch_size = batch_window.size(0)
        x, y = (
            batch_window[:, 0 : -self.prediction_length, :],
            batch_window[:, -self.prediction_length :, self.prediction_dims],
        )

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])

        recst = self.predcitor(lstm_out).view(
            self.batch_size, self.prediction_length, len(self.prediction_dims)
        )

        loss = self.loss_fn(recst, y)
        return_dict = {
            "loss": loss.sum(),
            "recst": recst,
            "score": loss,
            "y": y,
        }

        return return_dict

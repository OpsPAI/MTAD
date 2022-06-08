# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import logging
import time
import torch
from common.utils import set_device
from collections import defaultdict


class TimeSeriesEncoder(torch.nn.Module):
    def __init__(
        self,
        save_path,
        nb_epoch,
        lr,
        device="cpu",
        architecture="base",
        **kwargs,
    ):
        super().__init__()
        self.device = set_device(device)
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.best_metric = float("inf")
        self.time_tracker = {}
        self.model_save_file = os.path.join(save_path, f"{architecture}_model.ckpt")

    def compile(self):
        logging.info("Compiling finished.")
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.001
        )
        self = self.to(self.device)

    def save_encoder(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_encoder(self, model_save_path=""):
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(self.model_save_file, map_location=self.device))

    def fit(
        self,
        train_iterator,
        patience=10,
        **kwargs,
    ):
        num_batches = len(train_iterator)
        logging.info("Start training for {} batches.".format(num_batches))
        train_start = time.time()
        # Encoder training
        for epoch in range(1, self.nb_epoch + 1):
            running_loss = 0
            for idx, batch in enumerate(train_iterator):
                # batch: b x d x dim
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                self.optimizer.zero_grad()
                loss = return_dict["loss"]
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / num_batches
            logging.info("Epoch: {}, loss: {:.5f}".format(epoch, avg_loss))
            stop_training = self.__on_epoch_end(avg_loss, patience=patience)
            if stop_training:
                logging.info("Early stop at epoch {}.".format(epoch))
                break
        train_end = time.time()

        self.time_tracker["train"] = train_end - train_start
        return self

    def __on_epoch_end(self, monitor_value, patience):
        if monitor_value < self.best_metric:
            self.best_metric = monitor_value
            logging.info("Saving model for performance: {:.3f}".format(monitor_value))
            self.save_encoder()
            self.worse_count = 0
        else:
            self.worse_count += 1
        if self.worse_count >= patience:
            return True
        return False

    def encode(self, iterator):
        # Check if the given time series have unequal lengths
        save_dict = defaultdict(list)
        self = self.eval()

        used_keys = ["recst", "y", "diff"]
        with torch.no_grad():
            for batch in iterator:
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                for k in used_keys:
                    save_dict[k].append(return_dict[k])
        self = self.train()
        return {k: torch.cat(v) for k, v in save_dict.items()}

    def predict_prob(self, iterator, window_labels=None):
        logging.info("Evaluating")
        self = self.eval()
        test_start = time.time()
        with torch.no_grad():
            score_list = []
            for batch in iterator:
                batch = batch.to(self.device).float()
                return_dict = self(batch)
                score = (
                    # average all dimension
                    return_dict["score"]
                    .mean(dim=-1)
                    .sigmoid()  # b x prediction_length
                )
                # mean all timestamp
                score_list.append(score.mean(dim=-1))
        test_end = time.time()
        self.time_tracker["test"] = test_end - test_start

        anomaly_score = torch.cat(score_list, dim=0).cpu().numpy()
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_score, anomaly_label
        return anomaly_score

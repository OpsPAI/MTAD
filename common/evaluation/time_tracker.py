import time


class TimeTracker:
    def __init__(self, nb_epoch=1):
        self.train_time = 0
        self.test_time = 0
        self.nb_epoch = nb_epoch

    def train_start(self):
        self.s_train = time.time()

    def train_end(self):
        self.e_train = time.time()
        self.train_time = self.e_train - self.s_train

    def test_start(self):
        self.s_test = time.time()

    def test_end(self):
        self.e_test = time.time()
        self.test_time = self.e_test - self.s_test

    def get_data(self):
        return {
            "train_time": self.train_time,
            "test_time": self.test_time,
            "nb_epoch": self.nb_epoch,
        }

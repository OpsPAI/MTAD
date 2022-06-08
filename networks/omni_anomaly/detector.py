import numpy as np
import tensorflow as tf
from tfsnippet import VariableSaver
from tfsnippet.examples.utils import MLResults
from tfsnippet.utils import get_variables_as_dict, Config
from .model import OmniAnomaly
from .prediction import Predictor
from .training import Trainer
from tensorflow.python.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(
        self,
        data_array,
        batch_size=32,
        shuffle=False,
    ):
        self.darray = data_array
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_pool = list(range(self.darray.shape[0]))
        self.length = int(np.ceil(len(self.index_pool) * 1.0 / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        indexes = self.index_pool[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X = self.darray[indexes]

        # in case on_epoch_end not be called automatically :)
        if index == self.length - 1:
            self.on_epoch_end()
        return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_pool)


class ExpConfig(Config):
    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = "GRU"  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = "nf"  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 1
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 256
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.0
    bf_search_max = 400.0
    bf_search_step_size = 1.0

    valid_step_freq = 100
    gradient_clip_norm = 10.0

    early_stop = False  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.07

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = "model"
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = "result"  # Where to save the result file
    train_score_filename = "train_score.pkl"
    test_score_filename = "test_score.pkl"


class OmniDetector:
    def __init__(self, dim, model_root, window_size, initial_lr, l2_reg):
        self.config = self.__init_config(dim, model_root, window_size, initial_lr, l2_reg)
        self.time_tracker = {}
        self.__init_model()

    def __init_config(self, dim, model_root, window_size, initial_lr, l2_reg):
        config = ExpConfig()
        config.x_dim = dim
        config.result_dir = model_root
        config.window_length = window_size
        config.initial_lr = initial_lr
        config.l2_reg = l2_reg

        results = MLResults(config.result_dir)
        results.save_config(config)
        results.make_dirs(config.save_dir, exist_ok=True)
        return config

    def __init_model(self):
        tf.reset_default_graph()
        with tf.variable_scope("model") as model_vs:
            model = OmniAnomaly(config=self.config, name="model")
            # construct the trainer
            self.trainer = Trainer(
                model=model,
                model_vs=model_vs,
                max_epoch=self.config.max_epoch,
                batch_size=self.config.batch_size,
                valid_batch_size=self.config.test_batch_size,
                initial_lr=self.config.initial_lr,
                lr_anneal_epochs=self.config.lr_anneal_epoch_freq,
                lr_anneal_factor=self.config.lr_anneal_factor,
                grad_clip_norm=self.config.gradient_clip_norm,
                valid_step_freq=self.config.valid_step_freq,
            )

            # construct the predictor
            self.predictor = Predictor(
                model,
                batch_size=self.config.batch_size,
                n_z=self.config.test_n_z,
                last_point_only=True,
            )

    def fit(self, iterator):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.variable_scope("model") as model_vs:
            with tf.Session(config=tf_config).as_default():
                if self.config.restore_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = VariableSaver(
                        get_variables_as_dict(model_vs), self.config.restore_dir
                    )
                    saver.restore()

                best_valid_metrics = self.trainer.fit(iterator)

                self.time_tracker["train"] = best_valid_metrics["total_train_time"]
                if self.config.save_dir is not None:
                    # save the variables
                    var_dict = get_variables_as_dict(model_vs)
                    saver = VariableSaver(var_dict, self.config.save_dir)
                    saver.save()
                print("=" * 30 + "result" + "=" * 30)

    def predict_prob(self, iterator, label_windows=None):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.variable_scope("model") as model_vs:
            with tf.Session(config=tf_config).as_default():
                if self.config.save_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = VariableSaver(
                        get_variables_as_dict(model_vs), self.config.save_dir
                    )
                    saver.restore()

                score, z, pred_time = self.predictor.get_score(iterator)
                self.time_tracker["test"] = pred_time
        if label_windows is not None:
            anomaly_label = (np.sum(label_windows, axis=1) >= 1) + 0
            return score, anomaly_label
        else:
            return score

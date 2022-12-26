import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing

class DataFactory:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.home_dir = args.home_dir
        self.logger.info(f"current location: {os.getcwd()}")
        self.logger.info(f"home dir: {args.home_dir}")

        self.dataset_fn_dict = {
            "NeurIPS-TS-MUL": self.load_NeurIPS_TS_MUL,
            "SWaT": self.load_SWaT,
            "SMAP": self.load_SMAP,
            "MSL": self.load_MSL,
        }
        self.datasets = {
            "NeurIPS-TS-MUL": TSADStandardDataset,
            "SWaT": TSADStandardDataset,
            "SMAP": TSADStandardDataset,
            "MSL": TSADStandardDataset,
        }

        self.transforms = {
            "minmax": preprocessing.MinMaxScaler(),
            "std": preprocessing.StandardScaler(),
        }

    def __call__(self):
        self.logger.info(f"Preparing {self.args.dataset} ...")
        train_x, train_y, test_x, test_y = self.load()
        self.logger.info(
            f"train: X - {train_x.shape}, y - {train_y.shape} " +
            f"test: X - {test_x.shape}, y - {test_y.shape}"
        )
        self.logger.info(f"Complete.")

        self.logger.info(f"Preparing dataloader...")
        train_dataset, train_loader, test_dataset, test_loader = self.prepare(
            train_x, train_y, test_x, test_y,
            window_size=self.args.window_size,
            stride=self.args.stride,
            dataset_type=self.args.dataset,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            train_shuffle=True,
            test_shuffle=False,
            scaler=self.args.scaler,
            window_anomaly=self.args.window_anomaly
        )

        sample_X, sample_y = next(iter(train_loader))
        self.logger.info(f"total train dataset- {len(train_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")

        sample_X, sample_y = next(iter(test_loader))
        self.logger.info(f"total test dataset- {len(test_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")
        self.logger.info(f"Complete.")

        return train_dataset, train_loader, test_dataset, test_loader

    def load(self):
        return self.dataset_fn_dict[self.args.dataset](self.home_dir)

    def prepare(self, train_x, train_y, test_x, test_y,
                window_size,
                stride,
                dataset_type,
                batch_size,
                eval_batch_size,
                train_shuffle,
                test_shuffle,
                scaler,
                window_anomaly,
                ):

        transform = self.transforms[scaler]
        train_dataset = self.datasets[dataset_type](train_x, train_y,
                                                    flag="train", transform=transform,
                                                    window_size=window_size,
                                                    stride=stride,
                                                    window_anomaly=window_anomaly,
                                                    )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
        )

        transform = train_dataset.transform
        test_dataset = self.datasets[dataset_type](test_x, test_y,
                                                   flag="test", transform=transform,
                                                   window_size=window_size,
                                                   stride=stride,
                                                   window_anomaly=window_anomaly,
                                                   )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            shuffle=test_shuffle,
        )


        return train_dataset, train_dataloader, test_dataset, test_dataloader

    @staticmethod
    def load_NeurIPS_TS_MUL(home_dir="."):

        base_dir = "data/NeurIPS-TS"
        normal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_mul_normal.csv"))
        abnormal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_mul_abnormal.csv"))

        train_X, train_y = normal.values[:, :-1], normal.values[:, -1]
        test_X, test_y = abnormal.values[:, :-1], abnormal.values[:, -1]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_SWaT(home_dir="."):
        base_dir = "data/SWaT"
        SWAT_TRAIN_PATH = os.path.join(home_dir, base_dir, 'SWaT_Dataset_Normal_v0.csv')
        SWAT_TEST_PATH = os.path.join(home_dir, base_dir, 'SWaT_Dataset_Attack_v0.csv')
        df_train = pd.read_csv(SWAT_TRAIN_PATH, index_col=0)
        df_test = pd.read_csv(SWAT_TEST_PATH, index_col=0)

        def process_df(df):
            for var_index in [item for item in df.columns if item != 'Normal/Attack']:
                df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
            df.reset_index(drop=True, inplace=True)
            df.fillna(method='ffill', inplace=True)

            x = df.values[:, :-1].astype(np.float32)
            y = (df['Normal/Attack'] == 'Attack').to_numpy().astype(int)
            return x, y

        train_X, train_y = process_df(df_train)
        test_X, test_y = process_df(df_test)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_SMAP(home_dir="."):
        base_dir = "data/SMAP"
        with open(os.path.join(home_dir, base_dir, "SMAP_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "SMAP_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "SMAP_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_MSL(home_dir="."):
        base_dir = "data/MSL"
        with open(os.path.join(home_dir, base_dir, "MSL_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "MSL_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "MSL_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)
        return train_X, train_y, test_X, test_y

class TSADStandardDataset(Dataset):
    def __init__(self, x, y, flag, transform, window_size, stride, window_anomaly):
        super().__init__()
        self.transform = transform
        self.len = (x.shape[0] - window_size) // stride + 1
        self.window_size = window_size
        self.stride = stride
        self.window_anomaly = window_anomaly

        x, y = x[:self.len*self.window_size], y[:self.len*self.window_size]
        self.x = self.transform.fit_transform(x) if flag == "train" else self.transform.transform(x)
        self.y = y

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        _idx = idx * self.stride
        label = self.y[_idx:_idx+self.window_size]
        X, y = self.x[_idx:_idx+self.window_size], (1 in label) if self.window_anomaly else label
        return X, y

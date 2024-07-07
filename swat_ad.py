"""This module contains all utility functions for the SWaT dataset."""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@contextmanager
def measure_time(label: str) -> Generator[None, None, None]:
    """Measures the time and prints it to the console.

    Args:
        label: A label to identifiy the measured time.

    Yields:
        None.
    """
    start_time = time.time()
    yield
    print(f"{label} took {time.time()-start_time:.3f} seconds")
    

# Disable pylint too many locals for better readability of the loading function.
def load_pandas_dataframes(  # pylint: disable=too-many-locals, too-complex
    ratio: float,
    normalize: bool,
    downsample: bool,
) -> Tuple[List[pandas.DataFrame], List[dict], List[pandas.DataFrame], List[dict]]:
    """Loads the dataset as pandas dataframes.

    Args:
        pad: Whether to use zero padding or not.
        normalize: Whether to normalize the data with standard scaler or not.

    Returns:
        The dataframes and labels for each sample.
    """
    with measure_time("loading normal data"):
        # Read the dataset file columns as pandas dataframe
        path_normal = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"SWAT"/"SWaT_Dataset_Normal_v0_final.csv"
        dataset_dataframe = pandas.read_csv(path_normal)

        dataset_dataframe = dataset_dataframe.drop(columns= " Timestamp") 
        dataset_dataframe.rename(columns={"Normal/Attack": "category"}, inplace= True)
    
    # divide train-test normal data
    if downsample:
        with measure_time("downsampling data"):
            iter_dataset = list(dataset_dataframe.columns)[:-1] #leave out category

            highest_freq = -1
            for column in iter_dataset:
                x = (dataset_dataframe[column]).to_numpy()
                x = x - np.mean(x) # adjust offset

                N = len(x)

                # fourier transform
                y = fft(x)

                # amplitudes and frequency axis
                amplitudes = 2.0/N * np.abs(y[:N//2])

                T = 1.0
                freq = fftfreq(N, T)[:N//2]

                # pick first 5 highest frequencies
                freq_desc_order = np.argsort(amplitudes)[::-1]

                # keep overall highest frequency 
                amp = amplitudes[freq_desc_order[:5]]
                idxs = np.nonzero(amp)[0]
                if idxs.size == 0:
                    continue

                signal_freq = max(freq[freq_desc_order[idxs]])
                highest_freq = max(highest_freq, signal_freq)

            # resampling period
            res_T = int(1/(highest_freq*5))

            # downsample data
            dataset_dataframe = dataset_dataframe.iloc[::res_T, :]

    with measure_time("extract train and test normal dfs and labels"):
        train_set, dfs_test = train_test_split(dataset_dataframe, train_size= ratio, shuffle= False)
        train_labels = train_set.pop("category").to_frame()
        train_labels = train_labels.to_dict(orient= "records")
        
    # load attack test data
    with measure_time("loading attack data"):
        path_atk = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"SWAT"/"SWaT_Dataset_Split_Attack_v0_final.csv"
        atk_df = pandas.read_csv(path_atk)
        atk_df = atk_df.drop(columns= " Timestamp")
        atk_df.rename(columns={"Normal/Attack": "category"}, inplace= True)

        if downsample:
            atk_df = atk_df.iloc[::res_T, :]

        test_set = pandas.concat([dfs_test, atk_df], ignore_index=True, join="outer")
        
    # divide labels_metadata from test_data
    with measure_time("extract attack test dfs and labels"):
        meta_data = ["category","Attack #","Start Time","End Time","Attack Point","Start State","Attack","Actual Change","Expected Impact or attacker intent","Unexpected Outcome"]
        test_labels = test_set[meta_data].copy(deep= True)
        test_labels = test_labels.fillna('').astype('str')
        test_labels = test_labels.to_dict(orient= "records")

        test_set = test_set.drop(columns= meta_data)

    if normalize:
        with measure_time("normalize"):
            scale = StandardScaler()
            # using training data only
            scale.fit(train_set)

            train_set = pandas.DataFrame(scale.transform(train_set), columns= train_set.columns)
            test_set = pandas.DataFrame(scale.transform(test_set), columns= test_set.columns)

    train_set = train_set.to_numpy() 
    test_set = test_set.to_numpy()

    # add empty column if #columns is odd, avoids error with inner network models
    if train_set.shape[1] % 2 == 1:
        train_set = np.c_[train_set, np.zeros((train_set.shape[0], 1))]
        test_set = np.c_[test_set, np.zeros((test_set.shape[0], 1))]

    return train_set, train_labels, test_set, test_labels

def load_torch_tensors(
    ratio: float,
    normalize: bool,
    downsample: bool,
) -> Tuple[List[torch.Tensor], List[dict], List[torch.Tensor], List[dict]]:
    """Loads the dataset as torch tensors.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        pad: Whether to use zero padding or not.

    Returns:
        The tensors and labels for each sample.
    """
    x_train, y_train, x_test, y_test = load_pandas_dataframes(
        ratio=ratio,
        normalize=normalize,
        downsample=downsample,
    )

    train_arrays = [torch.from_numpy(row).float() for row in x_train]
    test_arrays = [torch.from_numpy(row).float() for row in x_test]

    return train_arrays, y_train, test_arrays, y_test


class SwatADDataset(Dataset):
    """The voraus-AD dataset torch adapter."""

    def __init__(
        self,
        tensors: List[torch.Tensor],
        labels: List[dict],
    ):
        """Initializes the voraus-AD dataset.

        Args:
            tensors: The tensors for each sample.
            labels: The labels for each sample.
        """
        self.tensors = tensors
        self.labels = labels

        assert len(self.tensors) == len(self.labels), "Can not handle different label and array length."
        self.length = len(self.tensors)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, Dict]:
        """Access single dataset samples.

        Args:
            item: The sample index.

        Returns:
            The sample and labels.
        """
        
        return self.tensors[item], self.labels[item]


# Disable pylint since we need all the arguments here.
def load_torch_dataloaders(  # pylint: disable=too-many-locals
    dataset: Union[Path, str],
    ratio: float,
    batch_size: int,
    seed: int,
    columns: Union[List[str], Tuple],
    normalize: bool,
    downsample: bool,
    frequency_divider: int,
    train_gain: float,
) -> tuple[SwatADDataset, SwatADDataset, DataLoader, DataLoader]:
    """Loads the pepper-AD dataset (train and test) as torch data loaders and datasets.

    Args:
        dataset: The path to the dataset.
        batch_size: The batch size to use.
        seed: The seed o use for the dataloader random generator.
        columns: The colums to load.
        normalize: Whether to normalize the data with standard scaler or not.
        frequency_divider: Scale the dataset down by dropping every nth sample.
        train_gain: The factor of train samples to use.
        pad: Whether to use zero padding or not.

    Returns:
        The data loaders and datasets.
    """
    x_train, y_train, x_test, y_test = load_torch_tensors(
        ratio=ratio,
        normalize=normalize,
        downsample=downsample,
    )

    train_dataset = SwatADDataset(x_train, y_train)
    test_dataset = SwatADDataset(x_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle= False)

    


    return train_dataset, None, train_dataloader, test_dataloader

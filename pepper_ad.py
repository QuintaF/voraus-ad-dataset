"""This module contains all utility functions for the Pepper dataset."""

import time
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from random import sample
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
) -> Tuple[List[pandas.DataFrame], List[dict], List[pandas.DataFrame], List[dict]]:
    """Loads the dataset as pandas dataframes.

    Args:
        pad: Whether to use zero padding or not.

    Returns:
        The dataframes and labels for each sample.
    """
    
    with measure_time("loading normal data"):
        # Read the dataset file columns as pandas dataframe 
        path_normal = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"pepper_csv-20240612T123704Z-001"/"pepper_csv"/"normal.csv"
        dataset_dataframe = pandas.read_csv(path_normal, skiprows=[1])
        dataset_dataframe = dataset_dataframe.drop(columns= "timestamp") # (20206, 256)

        # normal contains 4705 nan values for WheelFRTemp and WheelBTemp
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_df = pandas.DataFrame(imp.fit_transform(dataset_dataframe))
        new_df.columns= dataset_dataframe.columns
        new_df.index= dataset_dataframe.index

    # divide train-test normal data
    with measure_time("extract train dfs and labels"):
        train_set, dfs_test = train_test_split(new_df, train_size= ratio, shuffle= False)
        dfs_test["category"] = "normal"

    # load attack test data
    with measure_time("loading attack data"):
        path_joint = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"pepper_csv-20240612T123704Z-001"/"pepper_csv"/"JointControl.csv"
        path_leds = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"pepper_csv-20240612T123704Z-001"/"pepper_csv"/"LedsControl.csv"
        path_wheels = Path("D:/AI")/"voraus-ad-dataset"/"Pepper & SWAT"/"pepper_csv-20240612T123704Z-001"/"pepper_csv"/"WheelsControl.csv"
        atk_joint = pandas.read_csv(path_joint, skiprows=[1])
        atk_leds = pandas.read_csv(path_leds, skiprows=[1])
        atk_wheels = pandas.read_csv(path_wheels, skiprows=[1])

        atk_joint["category"] = "joints"
        atk_leds["category"] = "leds"
        atk_wheels["category"] = "wheels"
        atk_joint = atk_joint.drop(columns= "timestamp")
        atk_leds = atk_leds.drop(columns= "timestamp")
        atk_wheels = atk_wheels.drop(columns= "timestamp")

    with measure_time("extract attack test dfs and labels"):
        test_set = pandas.concat([dfs_test, atk_joint, atk_leds, atk_wheels], ignore_index=True)
        test_labels = (test_set.pop("category")).to_frame().to_dict(orient="records")

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

    return train_set, test_set, test_labels

def load_torch_tensors(
    ratio: float,
    normalize: bool,
) -> Tuple[List[torch.Tensor], List[dict], List[torch.Tensor], List[dict]]:
    """Loads the dataset as torch tensors.

    Args:
        path: The path to the dataset.
        columns: The colums to load.
        pad: Whether to use zero padding or not.

    Returns:
        The tensors and labels for each sample.
    """
    x_train, x_test, y_test = load_pandas_dataframes(
        ratio=ratio,
        normalize=normalize,
    )

    train_arrays = [torch.from_numpy(row).float() for row in x_train]
    test_arrays = [torch.from_numpy(row).float() for row in x_test]

    return train_arrays, test_arrays, y_test


class PepperADDataset(Dataset):
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
    pad: bool = True,
) -> tuple[PepperADDataset, PepperADDataset, DataLoader, DataLoader]:
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
    x_train, x_test, y_test = load_torch_tensors(
        ratio= ratio,
        normalize= normalize,
    )

    y_train = torch.rand(len(x_train)) # only for dataloader, not for use

    train_dataset = PepperADDataset(x_train, y_train)
    test_dataset = PepperADDataset(x_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, None, train_dataloader, test_dataloader

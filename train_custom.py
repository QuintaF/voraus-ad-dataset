"""Contains the training of the normalizing flow model."""

import warnings
import random
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy
import pandas
import torch
import torch.backends.cudnn
from sklearn import metrics
from torch import optim

# for Configuration model
from configuration import Configuration
import torch.utils
import torch.utils.data

from voraus_ad import Signals
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
from pepper_ad import load_torch_dataloaders as pepper_data_loaders
from swat_ad import load_torch_dataloaders as swat_data_loaders

# argparser
from arg_parser import parse_args

# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Contains the configuration model description."""
    
# Define the training configuration and hyperparameters of the model.
configuration = Configuration(
    columns="machine",  #only for voraus dataset
    epochs=70,
    frequencyDivider=1, #only for voraus dataset
    trainGain=1.0,      #only for voraus dataset
    seed=177,
    batchsize=32,
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=8e-4,
    normalize=False,
    pad=True,           #only for voraus dataset
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)



# Disable pylint too-many-variables here for readability.
# The whole training should run in a single function call.
def train(load_torch_dataloaders: Callable, ratio: float, downsample: bool) -> List[Dict]:  # pylint: disable=too-many-locals
    """Trains the model with the paper-given parameters.

    Returns:
        The auroc (mean over categories) and loss per epoch.
    """
    # Load the dataset as torch data loaders.
    train_dataset, _, train_dl, test_dl = load_torch_dataloaders(
        dataset= None,
        ratio=ratio,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        downsample=downsample,
    )

    # Retrieve the shape of the data for the model initialization.
    shape = train_dataset.tensors[0].shape
    if len(shape) < 2:
        n_signals = shape[0]
        n_times = 1
    else:
        n_signals = train_dataset.tensors[0].shape[1]
        n_times = train_dataset.tensors[0].shape[0]

    # Initialize the model, optimizer and scheduler.
    model = NormalizingFlow((n_signals, n_times), configuration).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
    )

    training_results: List[Dict] = []
    # Iterate over all epochs.
    for epoch in range(configuration.epochs):
        # TRAIN THE MODEL.
        model.train()
        loss: float = 0
        for tensors, _ in train_dl:
            tensors = tensors.float().to(DEVICE)

            # Execute the forward and jacobian calculation.
            optimizer.zero_grad()
            if len(tensors.shape) <3 :
                tensors = tensors.view(tensors.shape[0], n_times, n_signals)
                latent_z, jacobian = model.forward(tensors.transpose(2, 1)) 
            else:
                latent_z, jacobian = model.forward(tensors.transpose(2, 1))
                
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))

            # Back propagation and loss calculation.
            batch_loss = get_loss(latent_z, jacobian)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        # Calculate the mean loss over all samples.
        loss = loss / len(train_dl)

        # VALIDATE THE MODEL.
        model.eval()
        with torch.no_grad():
            result_list: List[Dict] = []
            for _, (tensors, labels) in enumerate(test_dl):
                tensors = tensors.float().to(DEVICE)

                # Calculate forward and jacobian.
                if len(tensors.shape) <3 :
                    tensors = tensors.view(tensors.shape[0], n_times, n_signals)
                    latent_z, jacobian = model.forward(tensors.transpose(2, 1))
                else:
                    latent_z, jacobian = model.forward(tensors.transpose(2, 1))

                jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
                # Calculate the anomaly score per sample.
                loss_per_sample = get_loss_per_sample(latent_z, jacobian)

                # Append the anomaly score and the labels to the results list.
                for j in range(loss_per_sample.shape[0]):
                    result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                    result_labels.update(score=loss_per_sample[j].item())
                    result_list.append(result_labels)

        results = pandas.DataFrame(result_list)

        # Calculate AUROC per anomaly category.
        aurocs = []
        if DATASET == "pepper":
            for category in ["joints", "leds", "wheels"]:
                dfn = results[(results["category"] == category) | (results["category"] == "normal")]
                fpr, tpr, _ = metrics.roc_curve(dfn["category"], dfn["score"].values, pos_label=category)
                auroc = metrics.auc(fpr, tpr)
                aurocs.append(auroc)
        else:
            for atk_num in [i for i in range(1,42) if i not in [5,9,12,15,18]]: 
                dfn = results[(results["Attack #"] == float(atk_num).__str__()) | (results["category"] == "Normal")]
                fpr, tpr, _ = metrics.roc_curve(dfn["category"], dfn["score"].values, pos_label="Attack")
                auroc = metrics.auc(fpr, tpr)
                aurocs.append(auroc)

        # Calculate the AUROC mean over all categories.
        aurocs_array = numpy.array(aurocs)
        auroc_mean = aurocs_array.mean()
        training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": loss})
        print(f"Epoch {epoch:0>3d}: auroc(mean)={auroc_mean:5.3f}, loss={loss:.6f}")

        scheduler.step()

    if MODEL_PATH is not None:
        torch.save(model.state_dict(), MODEL_PATH)

    return training_results


if __name__ == "__main__":
    args = parse_args()
    configuration.seed = args.seed
    configuration.normalize = args.normalize

    # Make the training reproducible.
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed_all(configuration.seed)
    numpy.random.seed(configuration.seed)
    random.seed(configuration.seed)
    if DETERMINISTIC_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    DATASET = (args.dataset).lower()
    # output filse name

    MODEL_PATH: Optional[Path] = Path.cwd() / f"model_seed{args.seed}_ratio{args.ratio}_{DATASET}{'_norm' if args.normalize else ''}{'_downsample' if args.downsample else ''}.pth"
    print(f"Model path after training: {MODEL_PATH}")
    if DATASET == "pepper":
        train(pepper_data_loaders, args.ratio, args.downsample)

    elif DATASET == "swat":
        train(swat_data_loaders, args.ratio, args.downsample)
        
    elif DATASET == "voraus":
        warnings.warn("Use the default 'train.py' file to train on voraus dataset!")
    else:
        raise(ValueError("custom dataset not accepted"))

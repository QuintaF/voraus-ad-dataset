
import random
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
from sklearn import metrics

from arg_parser import parse_args
from configuration import Configuration
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders as load_voraus
from pepper_ad import load_torch_dataloaders as pepper_data_loaders
from swat_ad import load_torch_dataloaders as swat_data_loaders
from metrics import compute_metrics

from normalizing_flow import NormalizingFlow, get_loss_per_sample


# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Define the training configuration and hyperparameters of the model.
configuration = Configuration(
    columns="machine",
    epochs=70,
    frequencyDivider=1,
    trainGain=1.0,
    seed=177,
    batchsize=32,
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=8e-4,
    normalize=True,
    pad=True,
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

def main():
    args = parse_args()
    configuration.seed = args.seed

    # Make the training reproducible.
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed_all(configuration.seed)
    np.random.seed(configuration.seed)
    random.seed(configuration.seed)
    if DETERMINISTIC_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = (args.dataset).lower()
    dataset_path = None
    if dataset == "pepper":
        load_fn = pepper_data_loaders
        configuration.normalize = args.normalize
        model_path: Optional[Path] = Path.cwd() / f"model_seed{args.seed}_ratio{args.ratio}_{dataset}{'_norm' if args.normalize else ''}{'_downsample' if args.downsample else ''}.pth"
        print(f"Loading model: {model_path}")
    elif dataset == "swat":
        load_fn = swat_data_loaders
        configuration.normalize = args.normalize
        model_path: Optional[Path] = Path.cwd() / f"model_seed{args.seed}_ratio{args.ratio}_{dataset}{'_norm' if args.normalize else ''}{'_downsample' if args.downsample else ''}.pth"
        print(f"Loading model: {model_path}")
    elif dataset == "voraus":
        model_path = Path.cwd() / "model.pth"
        print(f"Loading model: {model_path}")
        dataset_path = Path("D:/AI")/"voraus-ad-dataset"/"Dataset"/"voraus-ad-dataset-100hz.parquet"
        load_fn = load_voraus
    else:
        raise(ValueError("custom dataset not accepted"))
    
    if dataset == "voraus":
        # Load the dataset as torch data loaders.
        train_dataset, _, _, test_dl = load_fn(
        dataset=dataset_path,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        pad=configuration.pad,
    )
    else:
        # Load the dataset as torch data loaders.
        train_dataset, _, _, test_dl = load_fn(
            dataset= dataset_path,
            ratio=args.ratio,
            batch_size=configuration.batchsize,
            columns=Signals.groups()[configuration.columns],
            seed=configuration.seed,
            frequency_divider=configuration.frequency_divider,
            train_gain=configuration.train_gain,
            normalize=configuration.normalize,
            downsample=args.downsample,
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
    model.load_state_dict(torch.load(model_path))

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

    results = pd.DataFrame(result_list)

    # Calculate AUROC per anomaly category.
    aurocs = []
    if dataset == "voraus":
        m = []
        for category in ANOMALY_CATEGORIES:
            #keeps Normal_operation data and anomaly data from a specific anomaly
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label= True) 
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            m.append(compute_metrics(dfn, fpr, tpr, auroc, "Voraus", category.name, thresholds, True, args.plot))

    elif dataset == "pepper":
        results.rename(columns={"category": "anomaly"}, inplace= True)
        m = []
        for category in ["joints", "leds", "wheels"]:
            dfn = results[(results["anomaly"] == category) | (results["anomaly"] == "normal")]
            fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=category)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            m.append(compute_metrics(dfn, fpr, tpr, auroc, "Pepper", category, thresholds, category, args.plot))
    else: # if swat
        results.rename(columns={"category": "anomaly"}, inplace= True)
        m = []
        for category in [i for i in range(1,42) if i not in [5,9,12,15,18]]:
            dfn = results[(results["Attack #"] == float(category).__str__()) | (results["anomaly"] == "Normal")]
            fpr, tpr, thresholds = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label="Attack")
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            m.append(compute_metrics(dfn, fpr, tpr, auroc, "SWaT", category, thresholds, "Attack", args.plot))

    # Calculate the AUROC mean over all categories.
    aurocs_array = np.array(aurocs)
    auroc_mean = aurocs_array.mean()
    print(f"auroc_mean: {auroc_mean}")

    # compute mean over metrics
    m = np.array(m, dtype= object)
    m = np.mean(m, axis=0)
    print(f"Mean precision: {m[0]:.2f}\nMean recall: {m[1]:.2f}\nMean accuracy: {m[2]:.2f}\nMean F1-score: {m[3]:.2f}")

if __name__ == "__main__":
    main()
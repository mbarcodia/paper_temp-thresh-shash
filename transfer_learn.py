import sys
import xarray as xr
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchinfo
import importlib as imp
import pandas as pd
import warnings
import argparse


from datamaker.data_generator import ClimateData
from datamaker.data_generator_transfer import ClimateDataTransfer

from trainer.transfer_trainer_trainer import TransferTrainer  # CHANGED TO TRANSFER

# from trainer.trainer import Trainer

from model.model import TorchModel
from utils import utils
import model.loss as module_loss
import model.metric as module_metric
import visuals.plots as plots
from shash.shash_torch import Shash
import datamaker.data_loader as data_loader

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# print(f"python version = {sys.version}")
# print(f"numpy version = {np.__version__}")
# print(f"xarray version = {xr.__version__}")
# print(f"pytorch version = {torch.__version__}")

# --------------------------------------------------------


def make_transfer_model(transfer_model, trainable_ids, verbose=True):
    """
    Freezes all layers except those that contain any of the specified trainable_ids in their name.

    Args:
        transfer_model (torch.nn.Module): The model to modify.
        trainable_ids (list of str): List of substrings; layers containing these will remain trainable.
        verbose (bool): Whether to print layer statuses.

    Returns:
        torch.nn.Module: The modified model with frozen/unfrozen layers.
    """
    for name, param in transfer_model.named_parameters():
        param.requires_grad = any(
            trainable_id in name for trainable_id in trainable_ids
        )
        if param.requires_grad:
            print(f"Optimizing: {name}")

    if verbose:
        print("Final layer freeze status:")
        for name, param in transfer_model.named_parameters():
            print(f"Layer: {name}, Trainable: {param.requires_grad}")

        print("Final model architecture after freezing:")
        print(transfer_model)

    return transfer_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    args = parser.parse_args()
    config_tf = utils.get_config(args.expname)
    # Load the config for the pretrained model
    config_base = utils.get_config(config_tf["expname_base"])

    # Loop through random seeds
    for seed in config_tf["seed_list"]:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # Get the Data for Base and Transfer
        print("___________________")
        print("Get the data.")
        data_tf = ClimateDataTransfer(
            config_tf["datamaker"],
            expname=config_tf["expname"],
            seed=seed,
            data_dir=config_tf["data_dir"],
            figure_dir=config_tf["figure_dir"],
            verbose=False,
        )

        trainset_tf = data_loader.CustomData(data_tf.d_train)
        obsset_tf = data_loader.CustomData(data_tf.d_obs)

        data_base = ClimateData(
            config_base["datamaker"],
            expname=config_base["expname"],
            seed=seed,
            data_dir=config_base["data_dir"],
            figure_dir=config_base["figure_dir"],
            verbose=False,
        )

        trainset_base = data_loader.CustomData(data_base.d_train)
        valset_base = data_loader.CustomData(data_base.d_val)
        testset_base = data_loader.CustomData(data_base.d_test)
        obsset_base = data_loader.CustomData(data_base.d_obs)

        # Load the pretrained model on GCMs
        print("Loading the pretrained model for transfer learning.")

        device = utils.prepare_device(config_tf["device"])
        model_name_base = utils.get_model_name(config_tf["expname_base"], seed)
        print("___________________")
        print(f"Pretrained model name: {model_name_base}")

        # Load base model architecture and pretrained weights
        model_base = TorchModel(
            config=config_base["arch"],
            target=trainset_base.target,  # rescalers based on the original training data
        )  # sets up model architecture

        model_base = utils.load_torch_model(
            model_base, config_base["model_dir"] + model_name_base + ".pt"
        )  # loads model weights
        # model_base.to(device)  # Send to GPU/CPU

        print(f"Loaded pretrained model: {model_name_base}")

        # Apply layer freezing for transfer learning
        print("Fine-tuning model with observations")

        trainable_layers = [
            "denseblock_mu.0.0",  # First dense layer in denseblock_mu
            "denseblock_mu.1.0",  # Second dense layer in denseblock_mu
            "denseblock_mu.2.0",  # Third dense layer in denseblock_mu
            "finaldense_mu.0",  # Final dense layer in mu
            "output_mu",  # Output layer for mu
            # "denseblock_tau.0.0",  # First dense layer in denseblock_mu
            # "denseblock_tau.1.0",  # Second dense layer in denseblock_mu
            # "denseblock_tau.2.0",  # Third dense layer in denseblock_mu
            # "finaldense_tau.0",  # Final dense layer in mu
            # "output_tau",  # Output layer for mu
            # "denseblock_gamma.0.0",  # First dense layer in denseblock_mu
            # "denseblock_gamma.1.0",  # Second dense layer in denseblock_mu
            # "denseblock_gamma.2.0",  # Third dense layer in denseblock_mu
            # "finaldense_gamma.0",  # Final dense layer in mu
            # "output_gamma",  # Output layer for mu
            # "denseblock_sigma.0.0",  # First dense layer in denseblock_mu
            # "denseblock_sigma.1.0",  # Second dense layer in denseblock_mu
            # "denseblock_sigma.2.0",  # Third dense layer in denseblock_mu
            # "finaldense_sigma.0",  # Final dense layer in mu
            # "output_sigma",  # Output layer for mu
        ]  # Specify multiple trainable layer names

        model_transfer = make_transfer_model(
            model_base, trainable_ids=trainable_layers, verbose=True
        )

        # model.freeze_layers(freeze_id="tau")
        optimizer = getattr(torch.optim, config_tf["optimizer"]["type"])(
            model_transfer.parameters(), **config_tf["optimizer"]["args"]
        )

        # Define loss function and metrics using transfer learning config
        criterion = getattr(module_loss, config_tf["criterion"])()
        metric_funcs = [getattr(module_metric, met) for met in config_tf["metrics"]]

        # Learning rate scheduler (from transfer learning config)
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
        # )
        scheduler = StepLR(
            optimizer, step_size=250, gamma=0.5
        )  # Reduce LR every 50 epochs

        # Create DataLoader for the observational data
        obs_loader = torch.utils.data.DataLoader(
            obsset_tf,
            batch_size=config_tf["datamaker"]["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        # Rebuild the trainer for the observational data
        trainer = TransferTrainer(
            model_transfer,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs=config_tf["trainer"]["max_epochs"],
            data_loader=obs_loader,  # Use observational data loader
            validation_data_loader=None,
            device=device,
            config=config_tf,
            scheduler=scheduler,
            # early_stopping_metric=early_stopping_metric,  # Pass chosen metric
            # early_stopping_patience=config["trainer"]["early_stopping_patience"],
        )

        # for param_group in optimizer.param_groups:
        #     print(f"Current Learning Rate: {param_group['lr']}")

        # Visualize the model
        torchinfo.summary(
            model_transfer,
            [
                obsset_tf.input[: config_tf["datamaker"]["batch_size"]].shape,
                obsset_tf.input_unit[: config_tf["datamaker"]["batch_size"]].shape,
                obsset_tf.input_co2[: config_tf["datamaker"]["batch_size"]].shape,
            ],
            verbose=0,
            col_names=("input_size", "output_size", "num_params"),
        )

        # Train the Model
        model_transfer.to(device)
        trainer.fit()
        model_transfer.eval()

        # Save the Transfer Learned Model
        model_name_transfer = (
            model_name_base + "_tf" + config_tf["expname"] + "_transfer_learn"
        )
        utils.save_torch_model(
            model_transfer, config_tf["model_dir"] + model_name_transfer
        )

        print(f"Saved Transfer Learned Model: {model_name_transfer}")

        # Compute metrics and make visualizations
        print("___________________")
        print("Computing metrics and assessing the model predictions.")

        # mask = np.isclose(
        #     data_base.d_obs["temp_target"], 0.5
        # )  # Validation data is the 0.5 threshold
        # # Step 3: Apply mask to filter all keys
        # data_tf.d_val = {key: value[mask] for key, value in data_base.d_obs.items()}

        # Step 4: Print final filtered result
        # print("Filtered Data:", val_data)

        # output_test = output_val  # Placeholder for now
        data_tf.d_val = data_tf.d_obs  # Placeholder for now
        data_tf.d_test = data_tf.d_val  # Placeholder for now

        # Make predictions for train/val/test
        with torch.inference_mode():
            output_val = model_transfer.predict(
                dataset=obsset_tf, batch_size=128, device=device
            )
            output_test = model_transfer.predict(
                dataset=obsset_tf, batch_size=128, device=device
            )

        print(
            f"Output shape: {output_val.shape}, Target shape: {data_tf.d_val['y'].shape}"
        )

        # Compute and save the final metrics
        error_val = module_metric.custom_mae(output_val, data_tf.d_val["y"])
        error_test = module_metric.custom_mae(output_test, data_tf.d_test["y"])

        _, _, d_val, _ = module_metric.pit_d(output_val, data_tf.d_val["y"])
        _, _, d_test, _ = module_metric.pit_d(output_test, data_tf.d_test["y"])
        _, _, d_valtest, _ = module_metric.pit_d(
            np.append(output_val, output_test, axis=0),
            np.append(data_tf.d_val["y"], data_tf.d_test["y"], axis=0),
        )

        loss_val = float(criterion(output_val, data_tf.d_val["y"]).numpy())
        loss_test = float(criterion(output_test, data_tf.d_test["y"]).numpy())

        # fill and save the metrics dictionary
        d = {}
        d["exp_name"] = config_tf["expname"]
        d["rng_seed"] = seed
        d["error_val"] = error_val
        d["error_test"] = error_test
        d["loss_val"] = loss_val
        d["loss_test"] = loss_test
        d["d_val"] = d_val
        d["d_test"] = d_test
        d["d_valtest"] = d_valtest

        df = pd.DataFrame(d, index=[0]).reset_index(drop=True)
        df.to_pickle(config_tf["output_dir"] + model_name_transfer + "_metrics.pickle")

        # create and save diagnostics plots
        plots.plot_metrics_panels(trainer, config_tf)
        plots.savefig(
            config_tf["figure_dir"]
            + "model_diagnostics/"
            + model_name_transfer
            + "_metrics_diagnostic",
            fig_format=(".png",),
            dpi=config_tf["fig_dpi"],
        )
        plt.close()

        plots.plot_one_to_one_diagnostic_single(
            output_val,
            output_test,
            data_tf.d_val["y"],
            data_tf.d_test["y"],
            data_tf.d_test["year"],
        )
        plots.savefig(
            config_tf["figure_dir"]
            + "model_diagnostics/"
            + model_name_transfer
            + "_one_to_one_diagnostic",
            fig_format=(".png",),
            dpi=config_tf["fig_dpi"],
        )
        plt.close()

        plots.plot_pits(output_val, data_tf.d_val["y"])
        plots.savefig(
            config_tf["figure_dir"]
            + "model_diagnostics/"
            + model_name_transfer
            + "_pit",
            fig_format=(".png",),
            dpi=config_tf["fig_dpi"],
        )
        plt.close()

        # ----------------------------------------
        # PLOT INDIVIDUAL SSPS

        # Get unique SSPs
        unique_ssps = np.unique(data_tf.d_test["ssp"])

        # 1. Create Combined Figures
        # # Create a figure for metrics diagnostic
        # fig_metrics, axes_metrics = plt.subplots(
        #     len(unique_ssps), 4, figsize=(20, 4 * len(unique_ssps))
        # )
        # fig_metrics.suptitle("Metrics Diagnostic Across SSPs")

        # Create a figure for one-to-one diagnostics
        fig_one_to_one, axes_one_to_one = plt.subplots(
            len(unique_ssps), 2, figsize=(15, 6 * len(unique_ssps))
        )
        fig_one_to_one.suptitle("One-to-One Diagnostic Across SSPs")

        # Create a figure for PITs
        fig_pit, axes_pit = plt.subplots(
            len(unique_ssps), 1, figsize=(8, 5 * len(unique_ssps))
        )
        fig_pit.suptitle("PIT Across SSPs")

        # Ensure axes_pit is always iterable
        if isinstance(axes_pit, plt.Axes):  # If it's a single Axes, wrap it in a list
            axes_pit = [axes_pit]

        # Loop through each SSP
        for idx, ssp in enumerate(unique_ssps):
            # Create masks for the current SSP
            ssp_mask_val = data_tf.d_val["ssp"] == ssp
            ssp_mask_test = data_tf.d_test["ssp"] == ssp
            print(f"SSP {ssp} has {np.sum(ssp_mask_val)} validation samples.")
            print(f"SSP {ssp} has {np.sum(ssp_mask_test)} test samples.")

            # Filter the outputs and targets
            output_val_ssp = output_val[ssp_mask_val]
            target_val_ssp = data_tf.d_val["y"][ssp_mask_val]
            output_test_ssp = output_test[ssp_mask_test]
            target_test_ssp = data_tf.d_test["y"][ssp_mask_test]

            # Extract yrs_test for the current SSP
            yrs_test_ssp = data_tf.d_test["year"][ssp_mask_test]

            # Compute metrics for the filtered SSP data
            error_val = module_metric.custom_mae(output_val_ssp, target_val_ssp)
            error_test = module_metric.custom_mae(output_test_ssp, target_test_ssp)

            _, _, d_val, _ = module_metric.pit_d(output_val_ssp, target_val_ssp)
            _, _, d_test, _ = module_metric.pit_d(output_test_ssp, target_test_ssp)

            # # Since these metrics are computed during training, they can't be computed for individual SSPs without altering the trainer, so I will leave for now.

            # plots.plot_metrics_panels(
            #     trainer,
            #     config,
            #     fig=fig_metrics,
            #     axes=axes_metrics[idx],
            #     title=f"SSP {ssp}",
            # )

            # Plot one-to-one diagnostic for each SSP
            plots.plot_one_to_one_diagnostic(
                output_val_ssp,
                output_test_ssp,
                target_val_ssp,
                target_test_ssp,
                yrs_test_ssp,
                # data.d_test["year"][ssp_mask_test],
                ax=axes_one_to_one[idx],  # Specify the correct axes to plot to
                ssp_label=ssp,
            )

            # Plot PIT for each SSP
            plots.plot_pits(
                output_val_ssp,
                target_val_ssp,
                ax=axes_pit[idx],
                ssp_label=ssp,
            )  # Specify the correct axes

        # # Save the combined figures
        # fig_metrics.savefig(
        #     f"{config['figure_dir']}model_diagnostics/{model_name}_combined_metrics_diagnostic.png",
        #     # fig_format=(".png",),
        #     dpi=config["fig_dpi"],
        # )
        # plt.close(fig_metrics)

        fig_one_to_one.savefig(
            f"{config_tf['figure_dir']}model_diagnostics/{model_name_transfer}_combined_one_to_one_diagnostic.png",
            # fig_format=(".png",),
            dpi=config_tf["fig_dpi"],
        )
        plt.close(fig_one_to_one)

        fig_pit.savefig(
            f"{config_tf['figure_dir']}model_diagnostics/{model_name_transfer}_combined_pit.png",
            # fig_format=(".png",),
            dpi=config_tf["fig_dpi"],
        )
        plt.close(fig_pit)

        # ----------------------------------------
        print("Complete.")

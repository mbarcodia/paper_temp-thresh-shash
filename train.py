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
from trainer.trainer import Trainer
from model.model import TorchModel
import torch.nn as nn

from utils import utils
import model.loss as module_loss
import model.metric as module_metric
import visuals.plots as plots
from shash.shash_torch import Shash
import datamaker.data_loader as data_loader

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

# print(f"python version = {sys.version}")
# print(f"numpy version = {np.__version__}")
# print(f"xarray version = {xr.__version__}")
# print(f"pytorch version = {torch.__version__}")

# --------------------------------------------------------


def conv_block_shape(Nin, Cin, Hin, Win, channel_out, Kin, Sin, Pin):
    sample_tensor = torch.ones((Nin, Cin, Hin, Win))
    c2d = nn.Conv2d(
        in_channels=Cin,
        out_channels=channel_out,
        kernel_size=Kin,
        stride=Sin,
        padding=Pin,
    )
    maxpool = torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
    Nout, Cout, Hout, Wout = maxpool(c2d(sample_tensor)).detach().numpy().shape
    return Nout, Cout, Hout, Wout


def flatten_shape(Nin, Cin, Hin, Win):
    flat = torch.nn.Flatten(start_dim=1)
    sample_tensor = torch.ones(Nin, Cin, Hin, Win)
    Nout, Lout = flat(sample_tensor).detach().numpy().shape
    return Nout, Lout


def conv_flatten_shape(N, C, H, W, initial_pad, channel_out, Kin, Sin, Pin):
    # step through the architecture
    print("initial shape: " + str([N, C, H, W]))
    # padding
    W = W + initial_pad
    print("after padding: " + str([N, C, H, W]))
    # first convolution
    N, C, H, W = conv_block_shape(N, C, H, W, channel_out, Kin[0], Sin, Pin)
    print("after 1st conv block: " + str([N, C, H, W]))
    # flatten, this is the new output
    N, L = flatten_shape(N, C, H, W)
    print("after flattening conv block: " + str([N, L]))
    # Add the 2 scalars after flattening
    # L += 2
    # print(f"after adding scalar values: {L}")
    return N, C, H, W, L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    args = parser.parse_args()
    config = utils.get_config(args.expname)

    # Loop through random seeds
    for seed in config["seed_list"]:

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        model_name = utils.get_model_name(config["expname"], seed)
        print("___________________")
        print(model_name)

        # Get the Data
        print("___________________")
        print("Get the data.")
        data = ClimateData(
            config["datamaker"],
            expname=config["expname"],
            seed=seed,
            data_dir=config["data_dir"],
            figure_dir=config["figure_dir"],
            verbose=True,
        )

        trainset = data_loader.CustomData(data.d_train)
        valset = data_loader.CustomData(data.d_val)
        testset = data_loader.CustomData(data.d_test)

        print("valset size")
        print(valset.input.shape)
        print(valset.input_unit.shape)
        print(valset.input_co2.shape)
        print(valset.target.shape)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        # Get right shapes
        # # if self.config["hiddens_block_in"][0] == "auto":
        N, C, H, W, L = conv_flatten_shape(
            N=32,  # batch size
            C=config["arch"]["n_inputchannel"],  # channels
            H=72,  # height trainset.input.shape[1] lat
            W=144,  # weidth trainset.input.shape[2] lon
            initial_pad=config["arch"]["circular_padding"][0]
            + config["arch"]["circular_padding"][
                1
            ],  # needs both sides of padding in longitude direction
            channel_out=config["arch"]["filters"][-1],
            Kin=config["arch"]["kernel_size"],
            Sin=1,
            Pin="same",
        )
        print(f"Calculated flatten shape: {L}")
        print(f"Calculated output shape before flattening: {N}, {C}, {H}, {W}")
        # convolved and flattened => L

        # Update config manually after inspecting printed values
        # config["arch"]["hiddens_block_in"][0] = L
        # config["arch"]["hiddens_final_in"] = config["arch"]["hiddens_block"][-1] + 2

        print("used auto feature for building shapes of the model")
        print("new shape hiddens block in: " + str(config["arch"]["hiddens_block_in"]))
        print("new shape hiddens final in: " + str(config["arch"]["hiddens_final_in"]))

        # Setup the Model
        print("___________________")
        print("Building and training the model.")

        model = TorchModel(config=config["arch"], target=trainset.target)
        # model.freeze_layers(freeze_id="tau")
        optimizer = getattr(torch.optim, config["optimizer"]["type"])(
            model.parameters(), **config["optimizer"]["args"]
        )
        criterion = getattr(module_loss, config["criterion"])()
        metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]

        # Build the trainer
        device = utils.prepare_device(config["device"])
        trainer = Trainer(
            model,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs=config["trainer"]["max_epochs"],
            data_loader=train_loader,
            validation_data_loader=val_loader,
            device=device,
            config=config,
        )

        # Visualize the model
        # torchinfo.summary(
        #     model,
        #     [
        #         trainset.input[: config["datamaker"]["batch_size"]].shape,
        #         trainset.input_unit[: config["datamaker"]["batch_size"]].shape,
        #         trainset.input_co2[: config["datamaker"]["batch_size"]].shape,
        #     ],
        #     verbose=0,
        #     col_names=("input_size", "output_size", "num_params"),
        # )

        # Train the Model
        model.to(device)
        trainer.fit()
        model.eval()

        # Save the Pytorch Model
        utils.save_torch_model(model, config["model_dir"] + model_name + ".pt")

        # Compute metrics and make visualizations
        # Make and save cmip predictions for train/val/test
        print("___________________")
        print("Computing metrics and assessing the model predictions.")

        # Make predictions for train/val/test
        with torch.inference_mode():
            # output_train = model.predict(dataset=trainset, batch_size=128, device=device)
            output_val = model.predict(dataset=valset, batch_size=128, device=device)
            output_test = model.predict(dataset=testset, batch_size=128, device=device)

        # ----------------------------------------
        print(
            f"Output shape: {output_val.shape}, Target shape: {data.d_val['y'].shape}"
        )

        # Compute and save the final metrics
        error_val = module_metric.custom_mae(output_val, data.d_val["y"])
        error_test = module_metric.custom_mae(output_test, data.d_test["y"])

        _, _, d_val, _ = module_metric.pit_d(output_val, data.d_val["y"])
        _, _, d_test, _ = module_metric.pit_d(output_test, data.d_test["y"])
        _, _, d_valtest, _ = module_metric.pit_d(
            np.append(output_val, output_test, axis=0),
            np.append(data.d_val["y"], data.d_test["y"], axis=0),
        )

        loss_val = float(criterion(output_val, data.d_val["y"]).numpy())
        loss_test = float(criterion(output_test, data.d_test["y"]).numpy())

        # fill and save the metrics dictionary
        d = {}
        d["exp_name"] = config["expname"]
        d["rng_seed"] = seed
        d["error_val"] = error_val
        d["error_test"] = error_test
        d["loss_val"] = loss_val
        d["loss_test"] = loss_test
        d["d_val"] = d_val
        d["d_test"] = d_test
        d["d_valtest"] = d_valtest

        df = pd.DataFrame(d, index=[0]).reset_index(drop=True)
        df.to_pickle(config["output_dir"] + model_name + "_metrics.pickle")

        # create and save diagnostics plots
        plots.plot_metrics_panels(trainer, config)
        plots.savefig(
            config["figure_dir"]
            + "model_diagnostics/"
            + model_name
            + "_metrics_diagnostic",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        plots.plot_one_to_one_diagnostic_single(
            output_val,
            output_test,
            data.d_val["y"],
            data.d_test["y"],
            data.d_test["year"],
        )
        plots.savefig(
            config["figure_dir"]
            + "model_diagnostics/"
            + model_name
            + "_one_to_one_diagnostic",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        plots.plot_pits(output_val, data.d_val["y"])
        plots.savefig(
            config["figure_dir"] + "model_diagnostics/" + model_name + "_pit",
            fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close()

        # ----------------------------------------
        # PLOT INDIVIDUAL SSPS

        # Get unique SSPs
        unique_ssps = np.unique(data.d_test["ssp"])

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
            ssp_mask_val = data.d_val["ssp"] == ssp
            ssp_mask_test = data.d_test["ssp"] == ssp
            print(f"SSP {ssp} has {np.sum(ssp_mask_val)} validation samples.")
            print(f"SSP {ssp} has {np.sum(ssp_mask_test)} test samples.")

            # Filter the outputs and targets
            output_val_ssp = output_val[ssp_mask_val]
            target_val_ssp = data.d_val["y"][ssp_mask_val]
            output_test_ssp = output_test[ssp_mask_test]
            target_test_ssp = data.d_test["y"][ssp_mask_test]

            # Extract yrs_test for the current SSP
            yrs_test_ssp = data.d_test["year"][ssp_mask_test]

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
            f"{config['figure_dir']}model_diagnostics/{model_name}_combined_one_to_one_diagnostic.png",
            # fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close(fig_one_to_one)

        fig_pit.savefig(
            f"{config['figure_dir']}model_diagnostics/{model_name}_combined_pit.png",
            # fig_format=(".png",),
            dpi=config["fig_dpi"],
        )
        plt.close(fig_pit)

        # ----------------------------------------
        print("Complete.")

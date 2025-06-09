"""Metrics for generic plotting.

Functions
---------
plot_metrics(history,metric)
plot_metrics_panels(history, config)
plot_map(x, clim=None, title=None, text=None, cmap='RdGy')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as ct
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps as cmaps_ncl
import regionmask
import matplotlib.colors as mcolors
import gc
from scipy.optimize import curve_fit
from matplotlib import colors

from shash.shash_torch import Shash
import model.metric as module_metric


mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

FS = 10
plt.rc("text", usetex=False)
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("savefig", facecolor="white")
plt.rc("axes", facecolor="white")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("axes", labelcolor="dimgrey")
plt.rc("xtick", color="dimgrey")
plt.rc("ytick", color="dimgrey")


def savefig(filename, fig_format=(".png", ".pdf"), dpi=300):
    for fig_format in fig_format:
        plt.savefig(filename + fig_format, bbox_inches="tight", dpi=dpi)


def get_discrete_colornorm(cb_bounds, cmap):
    cb_n = int((cb_bounds[1] - cb_bounds[0]) / cb_bounds[-1])
    # cbar_n = (cb_bounds[1] - cb_bounds[-1]) - (cb_bounds[0] - cb_bounds[-1])
    clr_norm = colors.BoundaryNorm(
        np.linspace(
            cb_bounds[0] - cb_bounds[-1] / 2, cb_bounds[1] + cb_bounds[-1] / 2, cb_n + 2
        ),
        cmap.N,
    )

    return clr_norm


def plot_one_to_one_diagnostic_single(
    output_val,
    output_test,
    target_val,
    target_test,
    yrs_test,
):
    pr_val = Shash(output_val).median().numpy()
    pr_test = Shash(output_test).median().numpy()

    lowerbound_test = Shash(output_test).quantile(0.25).numpy()
    upperbound_test = Shash(output_test).quantile(0.75).numpy()

    mae_test = module_metric.custom_mae(output_test, target_test)

    # --------------------------------
    plt.subplots(1, 2, figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        target_val,
        pr_val,
        ".",
        label="validation",
        color="gray",
        alpha=0.75,
    )

    plt.errorbar(
        target_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        marker=".",
        linestyle="none",
        label="testing",
    )

    plt.axvline(x=0, color="gray", linewidth=1)
    plt.axhline(y=0, color="gray", linewidth=1)
    plt.title("Testing MAE = " + str(np.round(mae_test, 2)) + " yrs")
    plt.xlabel("target years until threshold is reached")
    plt.ylabel("predicted years until threshold is reached")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.errorbar(
        yrs_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        marker=".",
        linestyle="none",
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        label="testing",
    )

    plt.legend()
    plt.title("target years left until threshold is reached")
    plt.xlabel("year of map")
    plt.ylabel("predicted years until threshold is reached")
    plt.axhline(y=0, color="gray", linewidth=1)


def plot_one_to_one_diagnostic(
    output_val,
    output_test,
    target_val,
    target_test,
    yrs_test,
    ax=None,
    ssp_label="",
):
    if ax is None:
        ax = plt.gca()

    pr_val = Shash(output_val).median().numpy()
    pr_test = Shash(output_test).median().numpy()
    lowerbound_test = Shash(output_test).quantile(0.25).numpy()
    upperbound_test = Shash(output_test).quantile(0.75).numpy()
    mae_test = module_metric.custom_mae(output_test, target_test)

    # Check if ax is a single Axes object or an array of Axes
    if isinstance(ax, np.ndarray) and len(ax) > 1:
        ax1, ax2 = ax  # Two axes are expected in a subplot
    else:
        # Handle the single-axis case by reusing the same axis
        ax1, ax2 = ax, ax

    # Plot the first subplot (target vs. predicted) on ax1
    ax1.plot(target_val, pr_val, ".", label="validation", color="gray", alpha=0.75)
    ax1.errorbar(
        target_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        marker=".",
        linestyle="none",
        label="testing",
    )
    ax1.axvline(x=0, color="gray", linewidth=1)
    ax1.axhline(y=0, color="gray", linewidth=1)
    ax1.set_title(
        f"{ssp_label}    Testing MAE = " + str(np.round(mae_test, 2)) + " yrs"
    )
    ax1.set_xlabel("target years until threshold is reached")
    ax1.set_ylabel("predicted years until threshold is reached")
    ax1.legend()

    # Plot the second subplot (years vs. predicted years until threshold) on ax2
    ax2.errorbar(
        yrs_test,
        pr_test,
        yerr=np.concatenate(
            (
                pr_test - lowerbound_test[np.newaxis, :],
                upperbound_test[np.newaxis, :] - pr_test,
            ),
            axis=0,
        ),
        marker=".",
        linestyle="none",
        linewidth=0.5,
        color="tab:purple",
        alpha=0.5,
        label="testing",
    )
    ax2.legend()
    ax2.set_title("Target years left until threshold is reached")
    ax2.set_xlabel("Year of map")
    ax2.set_ylabel("Predicted years until threshold is reached")
    ax2.axhline(y=0, color="gray", linewidth=1)


# def plot_metrics_panels(trainer, config):
#     plt.figure(figsize=(20, 4))
#     for i, m in enumerate(("loss", *config["metrics"])):
#         plt.subplot(1, 4, i + 1)
#         plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
#         plt.plot(
#             trainer.log.history["epoch"],
#             trainer.log.history["val_" + m],
#             label="val_" + m,
#         )
#         plt.axvline(
#             x=trainer.early_stopper.best_epoch,
#             linestyle="--",
#             color="k",
#             linewidth=0.75,
#         )
#         plt.title(m)
#         plt.legend()
#         plt.xlabel("epoch")


def plot_metrics_panels(trainer, config, fig=None, axes=None, title=None):
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, m in enumerate(("loss", *config["metrics"])):
        ax = axes[i]
        ax.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
        # Check if 'val_' key exists in history and plot validation data- will be used for base model
        if "val_" + m in trainer.log.history:
            ax.plot(
                trainer.log.history["epoch"],
                trainer.log.history["val_" + m],
                label="val_" + m,
            )

        # check if "train_custom_mae" exists and plot it only for the MAE plot
        if m == config["metrics"][0] and "train_custom_mae" in trainer.log.history:
            ax.plot(
                trainer.log.history["epoch"],
                trainer.log.history["train_custom_mae"],
                label="train_custom_mae",
                linestyle="--",  # Optional: Customize the line style for clarity
            )

        # ax.plot(
        #     trainer.log.history["epoch"],
        #     trainer.log.history["val_" + m],
        #     label="val_" + m,
        # )
        # Only plot axvline if best_epoch exists
        if (
            hasattr(trainer.early_stopper, "best_epoch")
            and trainer.early_stopper.best_epoch is not None
        ):
            ax.axvline(
                x=trainer.early_stopper.best_epoch,
                linestyle="--",
                color="k",
                linewidth=0.75,
            )
        # ax.axvline(
        #     x=trainer.early_stopper.best_epoch,
        #     linestyle="--",
        #     color="k",
        #     linewidth=0.75,
        # )
        ax.set_title(m)
        ax.legend()
        ax.set_xlabel("epoch")

    if title:
        fig.suptitle(title)


def drawOnGlobe(
    ax,
    map_proj,
    data,
    lats,
    lons,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    inc=None,
    cbarBool=True,
    contourMap=[],
    contourVals=[],
    fastBool=False,
    extent="both",
):
    data_crs = ct.crs.PlateCarree()
    data_cyc, lons_cyc = add_cyclic_point(
        data, coord=lons
    )  # fixes white line by adding point#data,lons#ct.util.add_cyclic_point(data, coord=lons) #fixes white line by adding point
    data_cyc = data
    lons_cyc = lons

    #     ax.set_global()
    #     ax.coastlines(linewidth = 1.2, color='black')
    #     ax.add_feature(cartopy.feature.LAND, zorder=0, scale = '50m', edgecolor='black', facecolor='black')
    land_feature = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        facecolor="None",
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_feature(land_feature)
    #     ax.GeoAxes.patch.set_facecolor('black')

    if fastBool:
        image = ax.pcolormesh(lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap)
    #         image = ax.contourf(lons_cyc, lats, data_cyc, np.linspace(0,vmax,20),transform=data_crs, cmap=cmap)
    else:
        image = ax.pcolor(
            lons_cyc, lats, data_cyc, transform=data_crs, cmap=cmap, shading="auto"
        )

    if np.size(contourMap) != 0:
        contourMap_cyc, __ = add_cyclic_point(
            contourMap, coord=lons
        )  # fixes white line by adding point
        ax.contour(
            lons_cyc,
            lats,
            contourMap_cyc,
            contourVals,
            transform=data_crs,
            colors="fuchsia",
        )

    if cbarBool:
        cb = plt.colorbar(
            image, shrink=0.45, orientation="horizontal", pad=0.02, extend=extent
        )
        cb.ax.tick_params(labelsize=6)
    else:
        cb = None

    image.set_clim(vmin, vmax)

    return cb, image


def add_cyclic_point(data, coord=None, axis=-1):
    # had issues with cartopy finding utils so copied for myself

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError("The coordinate must be 1-dimensional.")
        if len(coord) != data.shape[axis]:
            raise ValueError(
                "The length of the coordinate does not match "
                "the size of the corresponding dimension of "
                "the data array: len(coord) = {}, "
                "data.shape[{}] = {}.".format(len(coord), axis, data.shape[axis])
            )
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError("The coordinate must be equally spaced.")
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError(
            "The specified axis does not correspond to an array dimension."
        )
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def plot_pits(output, target, ax=None, ssp_label=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Compute PIT and the D statistic
    bins, hist_shash, D_shash, EDp_shash = module_metric.pit_d(output, target)
    clr_shash = "teal"
    bins_inc = bins[1] - bins[0]

    # Calculate bin properties for the histogram
    bin_add = bins_inc / 2
    bin_width = bins_inc * 0.98

    # Plot histogram
    ax.bar(
        hist_shash[1][:-1] + bin_add,
        hist_shash[0],
        width=bin_width,
        color=clr_shash,
        label="SHASH",
    )

    # Add dashed line at y=0.1
    ax.axhline(y=0.1, linestyle="--", color="k", linewidth=2.0)

    # Customize y-ticks and y-limits specifically for this axis
    yticks = np.around(np.arange(0, 0.55, 0.05), 2)
    ax.set_yticks(yticks)
    ax.set_ylim(0, 0.25)

    # Customize x-ticks with bins
    ax.set_xticks(bins)
    ax.set_xticklabels(np.around(bins, 1))

    # Add the D statistic to the plot, including the SSP label
    ax.text(
        0.0,
        np.max(ax.get_ylim()) * 0.99,
        f"{ssp_label} - D statistic: {np.round(D_shash, 4)} (Expected: {np.round(EDp_shash, 3)})",
        color=clr_shash,
        verticalalignment="top",
        fontsize=12,
    )

    # Set labels
    ax.set_xlabel("Probability Integral Transform")
    ax.set_ylabel("Probability")


def plot_label_definition(
    year_reached,
    temp_target,
    global_mean_ens,
    global_mean,
    baseline_mean,
    anomalies,
    iyrs,
    config,
):

    # plot the calculation to make sure things make sense
    plt.figure(dpi=200)

    color = mcolors._colors_full_map
    color = list(color.values())
    rng = np.random.default_rng(seed=31415)
    rng.shuffle(color)

    for ens in np.arange(0, global_mean_ens.shape[0]):
        plt.plot(
            global_mean_ens["time.year"],
            global_mean_ens[ens, :],
            linewidth=1.0,
            color=color[ens],
            alpha=0.5,
        )
        plt.axvline(
            x=year_reached[ens],
            color=color[ens],
            linewidth=1.0,
            linestyle="-",
            # label=f"ens{ens}: {temp_target[ens].round(2)}C in {year_reached[ens]}",
            label=f"#{ens}: {temp_target[ens].round(2)}C in {year_reached[ens]}",
        )
    plt.plot(
        global_mean["time.year"],
        global_mean,
        linewidth=1,
        color="k",
        alpha=0.75,
    )

    plt.axhline(
        y=baseline_mean + temp_target[0],
        color="k",
        linestyle="--",
        linewidth=1,
    )
    plt.axhline(
        y=baseline_mean,
        color="k",
        linestyle="-",
        linewidth=1,
    )
    plt.legend(fontsize=6)
    # plt.tight_layout()


def plot_anomaly_definition(
    year_reached,
    temp_target,
    anomalies_ens,
    anomalies_mean,
    anomalies,
    iyrs,
    config,
):

    # plot the calculation to make sure things make sense
    plt.figure(dpi=200)

    color = mcolors._colors_full_map
    color = list(color.values())
    rng = np.random.default_rng(seed=31415)
    rng.shuffle(color)
    # for temp in temp_target:
    for ens in np.arange(0, anomalies.shape[0]):
        plt.plot(
            anomalies_ens["time.year"],
            anomalies[ens, :],
            linewidth=1.0,
            color=color[ens],
            alpha=0.5,
        )
        plt.axvline(
            x=year_reached[ens],
            color=color[ens],
            linewidth=1.0,
            linestyle="-",
            label=f"#{ens}: {temp_target[ens].round(2)}C in {year_reached[ens]}",
        )
    plt.plot(  # this should be the ensemble mean
        anomalies_ens["time.year"],
        anomalies_mean,
        linewidth=1,
        color="k",
        alpha=0.75,
    )

    plt.axhline(
        y=temp_target[0],
        color="k",
        linestyle="--",
        linewidth=1,
    )
    plt.axhline(
        # y=baseline_mean,
        y=0,  # solid black line at 0deg
        color="k",
        linestyle="-",
        linewidth=1,
    )

    plt.legend(fontsize=6)
    # plt.tight_layout()


def plot_single_anomaly_definition(
    year_reached,
    temp_target,
    anomalies_ens,
    anomalies,
    iyrs,
    config,
):

    # plot the calculation to make sure things make sense
    plt.figure(dpi=200)

    color = mcolors._colors_full_map
    color = list(color.values())
    rng = np.random.default_rng(seed=31415)
    rng.shuffle(color)
    for temp in temp_target:
        for ens in np.arange(0, 1):
            # ens = (0,)  # just the first ensemble member
            plt.plot(
                anomalies_ens["time.year"],
                anomalies[ens, :],
                linewidth=1.0,
                color=color[ens],
                alpha=0.5,
            )
            plt.axvline(
                x=year_reached[ens],
                color=color[ens],
                linewidth=1.0,
                linestyle="-",
                label=f"#{ens}: {temp_target[ens].round(2)}C in {year_reached[ens]}",
            )
            plt.plot(  # this should be the ensemble mean
                anomalies_ens["time.year"],
                anomalies[ens, :],
                linewidth=1,
                color="k",
                alpha=0.75,
            )

        plt.axhline(
            y=temp,
            color="k",
            linestyle="--",
            linewidth=1,
        )
        plt.axhline(
            # y=baseline_mean,
            y=0,  # solid black line at 0deg
            color="k",
            linestyle="-",
            linewidth=1,
        )

        plt.legend(fontsize=6)
        # plt.tight_layout()


################################
################################
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")

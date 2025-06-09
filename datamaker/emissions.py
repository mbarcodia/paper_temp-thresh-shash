"""Emissions functions for the SSPs

Functions
---------
get_emissions()
"""

__author__ = "Noah Diffenbaugh and Elizabeth A. Barnes"
__date__ = "24 January 2024"


import numpy as np
from matplotlib import pyplot as plt
import visuals.plots as plots
import pandas as pd
import datamaker.filemethods as filemethods


def create_emissions(ssp, years, data_dir, figure_dir, plot=False):

    emiss, emiss_years = get_emissions(ssp, data_dir, figure_dir, plot=False)
    cum_emiss_left = get_cumulative_emissions_left(emiss)

    if len(emiss_years) != len(years):
        istart = int(np.where(emiss_years == years[0])[0])
        iend = int(np.where(emiss_years == years[-1])[0] + 1)

        emiss_years = emiss_years[istart:iend]
        cum_emiss_left = cum_emiss_left[istart:iend]

    return emiss, cum_emiss_left, emiss_years


def get_cumulative_emissions_left(emiss):
    total_emiss = np.sum(emiss[emiss > 0])
    return total_emiss - np.cumsum(np.abs(emiss))


def get_emissions(ssp, data_dir, figure_dir, plot=False):

    # SSP EMISSIONS - cumulative emissions
    x_ssp = [2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
    x_interp = np.arange(1850, 2101)

    ssp119 = [
        39152.726,
        39693.726,
        22847.271,
        10475.089,
        2050.362,
        -1525.978,
        -4476.970,
        -7308.783,
        -10565.023,
        -13889.788,
    ]
    ssp126 = [
        39152.726,
        39804.013,
        34734.424,
        26509.183,
        17963.539,
        10527.979,
        4476.328,
        -3285.043,
        -8385.183,
        -8617.786,
    ]
    ssp245 = [
        39148.758,
        40647.530,
        43476.063,
        44252.900,
        43462.190,
        40196.485,
        35235.434,
        26838.373,
        16324.392,
        9682.859,
    ]
    ssp370 = [
        39148.758,
        44808.038,
        52847.359,
        58497.970,
        62904.059,
        66568.368,
        70041.979,
        73405.226,
        77799.049,
        82725.833,
    ]

    ssp585 = [
        39152.726,
        43712.349,
        55296.583,
        68775.698,
        83298.220,
        100338.606,
        116805.249,
        129647.035,
        130576.239,
        126287.310,
    ]

    # HISTORICAL EMISSIONS
    hist_filename, x_hist = filemethods.get_emissions_filename()
    iy = np.where((x_hist >= 1850) & (x_hist < x_ssp[0]))[0]
    hist = pd.read_csv(data_dir + hist_filename).to_numpy()[iy, 0] * 3.664 * 1000.0
    x_hist = x_hist[iy]

    # hist = [
    #     27962.868,
    #     28811.292,
    #     29658.679,
    #     33413.896,
    #     36131.477,
    #     37951.325,
    #     39628.028,
    # ]
    # x_hist = [1990, 1995, 2000, 2005, 2010, 2012, 2014]

    # CONCATENATE AND INTERPOLATE
    ssp119_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp119), axis=0),
        )
        * 0.001
    )
    ssp126_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp126), axis=0),
        )
        * 0.001
    )
    ssp245_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp245), axis=0),
        )
        * 0.001
    )
    ssp370_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp370), axis=0),
        )
        * 0.001
    )

    ssp585_interp = (
        np.interp(
            x_interp,
            np.concatenate((x_hist, x_ssp), axis=0),
            np.concatenate((hist, ssp585), axis=0),
        )
        * 0.001
    )

    i = np.where(ssp119_interp <= 0)[0]
    ssp119_yr = x_interp[i][0]

    i = np.where(ssp126_interp <= 0)[0]
    ssp126_yr = x_interp[i][0]

    # --------------------------------------------------------
    alpha = 0.8
    plot_emissions = False
    if plot_emissions:
        plt.figure(figsize=(7, 4))
        plt.axhline(y=0, color="dimgray", linewidth=1.0)

        plt.plot(
            x_interp,
            ssp119_interp,
            linewidth=3,
            color="teal",
            alpha=alpha,
            label="Hist + SSP1-1.9",
        )
        plt.plot(
            x_interp,
            ssp126_interp,
            linewidth=3,
            color="goldenrod",
            alpha=alpha,
            label="Hist + SSP1-2.6",
        )
        plt.plot(
            x_interp,
            ssp245_interp,
            linewidth=3,
            color="tab:purple",
            alpha=alpha,
            label="Hist + SSP2-4.5",
        )
        plt.plot(
            x_interp,
            ssp370_interp,
            linewidth=3,
            color="tab:pink",
            alpha=alpha,
            label="Hist + SSP3-7.0",
        )

        plt.plot(
            x_interp,
            ssp585_interp,
            linewidth=3,
            color="tab:pink",
            alpha=alpha,
            label="Hist + SSP5-8.5",
        )

        plt.legend()

        plt.annotate(
            ssp119_yr,
            (ssp119_yr, 0),
            color="teal",
            xytext=(2045, -7.5),
            arrowprops=dict(arrowstyle="->", color="teal", connectionstyle="arc3"),
        )

        plt.annotate(
            ssp126_yr,
            (ssp126_yr, 0),
            color="goldenrod",
            xytext=(2082, 5),
            arrowprops=dict(arrowstyle="->", color="gold", connectionstyle="arc3"),
        )

        plt.ylabel("Gt per year")
        plt.xlabel("year")

        # plots.format_spines(plt.gca())
        plt.xticks(np.arange(1850, 2150, 50), np.arange(1850, 2150, 50))
        plt.yticks(np.arange(-20, 100, 10), np.arange(-20, 100, 10))

        plt.xlim(1850, 2100)
        plt.ylim(-15.5, 85.5)

        plt.title("anthropogenic CO$_2$ emissions under Historical + SSPs")

        plots.savefig(figure_dir + "data_diagnostics/emissions")
        plt.close()

    # return emissions in Teratons
    if ssp == "ssp119":
        return ssp119_interp / 1000.0, x_interp
    elif ssp == "ssp126":
        return ssp126_interp / 1000.0, x_interp
    elif ssp == "ssp245":
        return ssp245_interp / 1000.0, x_interp
    elif ssp == "ssp370":
        return ssp370_interp / 1000.0, x_interp
    elif ssp == "ssp585":
        return ssp585_interp / 1000.0, x_interp
    else:
        raise NotImplementedError()

"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)
save_tf_model(model, model_name, directory, settings)
get_cmip_filenames(settings, verbose=0)
"""

import xarray as xr
import numpy as np
from collections import defaultdict


__author__ = (
    "original: Elizabeth A. Barnes and Noah Diffenbaugh; updated by Marybeth Arcodia"
)
__version__ = "16 July 2024"


def get_model_name(settings):
    model_name = settings["exp_name"] + "_rng_seed" + str(settings["rng_seed"])

    return model_name


def get_netcdf_da(filename):
    da = xr.open_dataarray(filename)
    return da


def get_additional_cmip6_model_names(directory):
    import glob

    modelnames = []
    filenames = glob.glob(directory + "/*")
    for f in filenames:
        k = f.find("historical_")
        m = f.find("_ann_mean")
        if k >= 0:
            modelnames.append(f[k + len("historical_") : m])
    return modelnames


def convert_to_cftime(da, orig_time):
    da = da.rename({orig_time: "time"})
    dates = xr.cftime_range(
        start="1850", periods=da.shape[0], freq="YS", calendar="noleap"
    )
    da = da.assign_coords(
        {"time": ("time", dates, {"units": "years since 1850-01-01"})}
    )
    return da


def get_gcm_model_names(directory):
    import glob

    modelnames = []
    filenames = glob.glob(directory + "/*")
    for f in filenames:
        k = f.find("historical_")
        m = f.find("_ann_mean")
        if k >= 0:
            modelnames.append(f[k + len("historical_") : m])
    return modelnames


def get_gcm_name(f, ssp):
    istart = f.find(ssp + "_") + 7
    iend = f.find("_r1-")
    return f[istart:iend]


def get_observations_filename_input(source, verbose=False):

    if source == "BEST":
        nc_filename_obs = (
            # "_Land_and_Ocean_LatLong1_185001_202312_ann_mean_2pt5degree.nc"
            "_Land_and_Ocean_LatLong1_185001_202412_ann_mean_2pt5degree.nc"
        )
    else:
        raise NotImplementedError()

    if verbose:
        print(nc_filename_obs)

    return nc_filename_obs


def get_observations_filename_output(source, verbose=False):

    if source == "BEST":
        nc_filename_obs = (
            # "tasmin_Land_ONLY_LatLong1_185001_202312_JJA_mean_2pt5degree.nc"
            "tasmin_Land_ONLY_LatLong1_185001_202410_JJA_mean_2pt5degree.nc"
        )
    else:
        raise NotImplementedError()

    if verbose:
        print(nc_filename_obs)

    return nc_filename_obs


def get_emissions_filename():
    """Returns emissions filename and years for the file."""
    return "global_carbon_budget_1750_2021.csv", np.arange(1750, 2022)


def get_cmip_filenames(ssp, var, period, sub, verbose=False):
    main_dict = filename_lookup_dict()
    if verbose:
        print("main_dict")
        print(main_dict)

    filenames = []
    filenames.extend(main_dict[ssp][var][period][sub])

    if verbose:
        print(filenames)

    return filenames


def filename_lookup_dict():
    main_dict = {}

    # #  -------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # SSP 1-2.6 - minimum temp JJA
    # --------------------------------------------------------------------------------------------
    d_child = {}
    d_child["multi_member"] = (
        "tasmin_Amon_hist_ssp126_ACCESS-CM2_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp126_ACCESS-ESM1-5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp126_CanESM5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp126_MIROC6_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp126_MPI-ESM1-2-LR_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp126" not in main_dict:
        main_dict["ssp126"] = {}
    if "tasmin" not in main_dict["ssp126"]:
        main_dict["ssp126"]["tasmin"] = {}
    if "jja" not in main_dict["ssp126"]["tasmin"]:
        main_dict["ssp126"]["tasmin"]["jja"] = {}
    main_dict["ssp126"]["tasmin"]["jja"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 2-4.5 - minimum temp JJA
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tasmin_Amon_hist_ssp245_ACCESS-CM2_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp245_ACCESS-ESM1-5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp245_CanESM5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp245_MIROC6_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp245_MPI-ESM1-2-LR_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp245" not in main_dict:
        main_dict["ssp245"] = {}
    if "tasmin" not in main_dict["ssp245"]:
        main_dict["ssp245"]["tasmin"] = {}
    if "jja" not in main_dict["ssp245"]["tasmin"]:
        main_dict["ssp245"]["tasmin"]["jja"] = {}
    main_dict["ssp245"]["tasmin"]["jja"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 3- 7.0 - - minimum temp JJA
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tasmin_Amon_hist_ssp370_ACCESS-CM2_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp370_ACCESS-ESM1-5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp370_CanESM5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp370_MIROC6_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp370_MPI-ESM1-2-LR_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp370" not in main_dict:
        main_dict["ssp370"] = {}
    if "tasmin" not in main_dict["ssp370"]:
        main_dict["ssp370"]["tasmin"] = {}
    if "jja" not in main_dict["ssp370"]["tasmin"]:
        main_dict["ssp370"]["tasmin"]["jja"] = {}
    main_dict["ssp370"]["tasmin"]["jja"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 5-8.5- JJA minimum temp
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tasmin_Amon_hist_ssp585_ACCESS-CM2_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp585_ACCESS-ESM1-5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp585_CanESM5_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp585_MIROC6_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
        "tasmin_Amon_hist_ssp585_MPI-ESM1-2-LR_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp585" not in main_dict:
        main_dict["ssp585"] = {}
    if "tasmin" not in main_dict["ssp585"]:
        main_dict["ssp585"]["tasmin"] = {}
    if "jja" not in main_dict["ssp585"]["tasmin"]:
        main_dict["ssp585"]["tasmin"]["jja"] = {}
    main_dict["ssp585"]["tasmin"]["jja"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 1-2.6- annual average temp
    # --------------------------------------------------------------------------------------------
    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_hist_ssp126_ACCESS-CM2_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp126_ACCESS-ESM1-5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp126_CanESM5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp126_MIROC6_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp126_MPI-ESM1-2-LR_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp126" not in main_dict:
        main_dict["ssp126"] = {}
    if "tas" not in main_dict["ssp126"]:
        main_dict["ssp126"]["tas"] = {}
    if "ANN" not in main_dict["ssp126"]["tas"]:
        main_dict["ssp126"]["tas"]["ANN"] = {}
    main_dict["ssp126"]["tas"]["ANN"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 2-4.5- annual average temp
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_hist_ssp245_ACCESS-CM2_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp245_ACCESS-ESM1-5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp245_CanESM5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp245_MIROC6_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp245_MPI-ESM1-2-LR_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp245" not in main_dict:
        main_dict["ssp245"] = {}
    if "tas" not in main_dict["ssp245"]:
        main_dict["ssp245"]["tas"] = {}
    if "ANN" not in main_dict["ssp245"]["tas"]:
        main_dict["ssp245"]["tas"]["ANN"] = {}
    main_dict["ssp245"]["tas"]["ANN"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 3-7.0 - annual average temp
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_hist_ssp370_ACCESS-CM2_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp370_ACCESS-ESM1-5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp370_CanESM5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp370_MIROC6_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp370_MPI-ESM1-2-LR_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp370" not in main_dict:
        main_dict["ssp370"] = {}
    if "tas" not in main_dict["ssp370"]:
        main_dict["ssp370"]["tas"] = {}
    if "ANN" not in main_dict["ssp370"]["tas"]:
        main_dict["ssp370"]["tas"]["ANN"] = {}
        main_dict["ssp370"]["tas"]["ANN"] = d_child
    main_dict["ssp370"]["tas"]["ANN"] = d_child

    # --------------------------------------------------------------------------------------------
    # SSP 5-8.5 - annual average temp
    # --------------------------------------------------------------------------------------------

    d_child = {}
    d_child["multi_member"] = (
        "tas_Amon_hist_ssp585_ACCESS-CM2_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp585_ACCESS-ESM1-5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp585_CanESM5_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp585_MIROC6_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
        "tas_Amon_hist_ssp585_MPI-ESM1-2-LR_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc",
    )

    d_child["single_member"] = ()

    if "ssp585" not in main_dict:
        main_dict["ssp585"] = {}
    if "tas" not in main_dict["ssp585"]:
        main_dict["ssp585"]["tas"] = {}
    if "ANN" not in main_dict["ssp585"]["tas"]:
        main_dict["ssp585"]["tas"]["ANN"] = {}
        main_dict["ssp585"]["tas"]["ANN"] = d_child
    main_dict["ssp585"]["tas"]["ANN"] = d_child

    return main_dict

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import cartopy as ct

from collections import defaultdict
import math


def compute_gridded_anomalies(da, config):
    """
    Compute gridded anomalies using the settings inside `config["datamaker"]`.

    Parameters:
        da (xarray.DataArray): A 4D dataset with dimensions (member, time, lat, lon).
        config (dict): Configuration dictionary with key "datamaker", which contains:
            - "anomalies" (bool or str)
            - "anomaly_yr_bounds" (tuple)
            - "baseline_yr_bounds" (tuple)
            - "remove_map_mean" (str)

    Returns:
        xarray.DataArray: Anomaly dataset.
    """

    datamaker = config["datamaker"]  # Access nested dictionary

    if datamaker["anomalies"] is True:
        da_anomalies = da - da.sel(
            time=slice(
                str(datamaker["anomaly_yr_bounds"][0]),
                str(datamaker["anomaly_yr_bounds"][1]),
            )
        ).mean("time")
        print("Computing anomalies with reference to anomaly_yr_bounds.")

    elif datamaker["anomalies"] == "baseline":
        da_anomalies = da - da.sel(
            time=slice(
                str(datamaker["baseline_yr_bounds"][0]),
                str(datamaker["baseline_yr_bounds"][1]),
            )
        ).mean("time")

        da_anomalies = da_anomalies - da_anomalies.sel(
            time=slice(
                str(datamaker["anomaly_yr_bounds"][0]),
                str(datamaker["anomaly_yr_bounds"][1]),
            )
        ).mean("time")
        print(
            "Computing anomalies with reference to baseline_yr_bounds then anomaly_yr_bounds."
        )

    elif not datamaker["anomalies"]:
        print("Not computing any anomalies...")
        return da  # Return original data

    else:
        raise NotImplementedError("Invalid anomalies setting in config.")

    # Remove spatial mean if specified
    if datamaker.get("remove_map_mean") == "raw":
        da_anomalies = da_anomalies - da_anomalies.mean(("lon", "lat"))
    elif datamaker.get("remove_map_mean") == "weighted":
        weights = np.cos(np.deg2rad(da_anomalies.lat))
        weights.name = "weights"
        da_anomalies_weighted = da_anomalies.weighted(weights)
        da_anomalies = da_anomalies - da_anomalies_weighted.mean(("lon", "lat"))

    return da_anomalies


def plot_multiple_storylines_min_temp(
    storyline_matches,
    d_test,
    d_obs,
    config,
    seed,
    num_years=1,
    lat_range=None,
    lon_range=None,
    vmin=-3,
    vmax=3,
    save=True,
):
    """
    Plots temperature maps for multiple SSPs in a 4-panel subplot.

    Parameters:
        storyline_matches (list): List of storyline matches with SSP, GCM, and member info.
        data (object): Data containing test dataset.
        config (object): Configuration for computing anomalies.
        lat_range (tuple): Latitude range for subsetting.
        lon_range (tuple): Longitude range for subsetting.
        projection alternative options: ccrs.EqualEarth(central_longitude=250)
    """

    num_ssps = len(storyline_matches)  # Get the number of SSPs
    num_cols = min(2, num_ssps)  # Use up to 2 columns
    num_rows = math.ceil(num_ssps / num_cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()  # Flatten in case of a single row or column

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tasmin_Amon_hist_{ssp}_{model}_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc"
        )

        # Compute anomalies
        da_anomalies = compute_gridded_anomalies(s1, config)

        # Select storyline based on test member
        storyline = da_anomalies[member, :, :, :]

        # Isolate median year of threshold crossing prediction
        # year_cross_med = storyline.sel(
        #     time=storyline["time"].dt.year == (year_obs_prediction + year_test_value)
        # )

        year_cross_med = storyline.sel(
            time=storyline["time"].dt.year.isin(
                range(
                    int(year_obs_prediction + year_test_value),
                    int(year_obs_prediction + year_test_value) + num_years,
                )
            )
        )

        year_cross_med = year_cross_med.mean("time")

        # Subset data if lat/lon range is provided
        if lat_range and lon_range:
            year_cross_med = year_cross_med.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        # Get lat/lon values
        lat = year_cross_med.lat.values
        lon = year_cross_med.lon.values

        # Set up colormap
        # cmap = cmaps_ncl.BlueDarkRed18.colors
        # cmap = plt.cm.get_cmap("RdBu_r")  # Use a diverging colormap
        # cmap = cm.get_cmap("Reds")

        # # norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # # Create a normalizer for discrete bins
        # bounds = np.round(np.linspace(vmin, vmax, vmax * 2 + 1), 2)  # Creates segments
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)

        cmap = cm.get_cmap("Reds")
        bounds = np.round(np.linspace(vmin, vmax, vmax * 4 + 1), 8)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot on subplot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )
        # ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", edgecolor="k", facecolor="None"))
        # Add country and state borders
        ax.coastlines(linewidth=1.5)  # Thicker country borders
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)  # International borders
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,  # Thicker state borders
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            year_cross_med[:, :],
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add IPCC region mask
        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar setup
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", shrink=0.8, extend="both", pad=0.05
    )
    cbar.set_label("Temperature Anomaly (°C)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Adjust font size of tick labels

    # Main title
    fig.suptitle(
        f"Storyline Analysis after Crossing the {temp_target}°C Regional Threshold \n Minimum JJA Temperature {num_years}-year Average",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()
    # plt.show()
    if save == True:
        plt.savefig(
            f"{config['figure_dir']}analysis/{config['expname']}_slines_minJJA_{temp_target}_{num_years}yravg_seed{seed}.png",
            dpi=300,
        )


def plot_multiple_storylines_ann_temp(
    storyline_matches,
    d_test,
    d_obs,
    config,
    seed,
    num_years=1,
    lat_range=None,
    lon_range=None,
    vmin=-3,
    vmax=3,
    save=True,
):
    """
    Plots temperature maps for multiple SSPs in a 4-panel subplot.

    Parameters:
        storyline_matches (list): List of storyline matches with SSP, GCM, and member info.
        data (object): Data containing test dataset.
        config (object): Configuration for computing anomalies.
        lat_range (tuple): Latitude range for subsetting.
        lon_range (tuple): Longitude range for subsetting.
        projection alternative options: ccrs.EqualEarth(central_longitude=250)
    """

    num_ssps = len(storyline_matches)  # Get the number of SSPs
    num_cols = min(2, num_ssps)  # Use up to 2 columns
    num_rows = math.ceil(num_ssps / num_cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()  # Flatten in case of a single row or column

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tas_Amon_hist_{ssp}_{model}_r1-10i1p1f1_ann_mean_gn_185001-210012_2pt5deg.nc"
        )

        # Compute anomalies
        da_anomalies = compute_gridded_anomalies(s1, config)

        # Select storyline based on test member
        storyline = da_anomalies[member, :, :, :]

        # Isolate constraint year of threshold crossing prediction
        # year_cross_med = storyline.sel(
        #     time=storyline["time"].dt.year == (year_obs_prediction + year_test_value)
        # )

        year_cross_med = storyline.sel(
            time=storyline["time"].dt.year.isin(
                range(
                    int(year_obs_prediction + year_test_value),
                    int(year_obs_prediction + year_test_value) + num_years,
                )
            )
        )

        year_cross_med = year_cross_med.mean("time")

        # Subset data if lat/lon range is provided
        if lat_range and lon_range:
            year_cross_med = year_cross_med.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        # Get lat/lon values
        lat = year_cross_med.lat.values
        lon = year_cross_med.lon.values

        # Set up colormap
        # cmap = cmaps_ncl.BlueDarkRed18.colors
        # cmap = plt.cm.get_cmap("RdBu_r")  # Use a diverging colormap
        # cmap = cm.get_cmap("Reds")
        # norm = plt.Normalize(vmin=vmin, vmax=vmax)

        cmap = cm.get_cmap("Reds")
        bounds = np.round(np.linspace(0, 5, 11), 2)  # Creates segments
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot on subplot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )
        # ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", edgecolor="k", facecolor="None"))
        # Add country and state borders
        ax.coastlines(linewidth=1.5)  # Thicker country borders
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)  # International borders
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,  # Thicker state borders
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            year_cross_med[:, :],
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add IPCC region mask
        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar setup
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", shrink=0.8, extend="both", pad=0.05
    )
    cbar.set_label("Temperature Anomaly (°C)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Adjust font size of tick labels

    # Main title
    fig.suptitle(
        f"Storyline Analysis after Crossing the {temp_target}°C Regional Threshold \n Annual Mean Temperature {num_years}-year Average",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout()
    # plt.show()
    if save == True:
        plt.savefig(
            f"{config['figure_dir']}analysis/{config['expname']}_slines_ann_mean_gn_{temp_target}_{num_years}yravg_seed{seed}.png",
            dpi=300,
        )


def plot_raw_temp_exceedance_count(
    storyline_matches,
    d_test,
    d_obs,
    config,
    seed,
    num_years=1,
    lat_range=None,
    lon_range=None,
    vmin=0,
    vmax=10,  # Adjust based on expected counts
    save=True,
):
    """
    Plots maps of the number of months (JJA) where temperature exceeds 294.2K.

    Parameters:
        storyline_matches (list): List of storyline matches with SSP, GCM, and member info.
        num_years (int): Number of years to compute the exceedance count.
        lat_range (tuple): Latitude range for subsetting.
        lon_range (tuple): Longitude range for subsetting.
    """

    num_ssps = len(storyline_matches)
    num_cols = min(2, num_ssps)
    num_rows = math.ceil(num_ssps / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tasmin_Amon_hist_{ssp}_{model}_r1-10i1p1f1_monthly_gn_185001-210012_2pt5deg.nc"
        )

        # Select the specified member
        storyline = s1[member, :, :, :]

        # Filter years
        selected_years = range(
            int(year_obs_prediction + year_test_value),
            int(year_obs_prediction + year_test_value) + num_years,
        )

        # Isolate JJA months and apply threshold
        year_cross_med = storyline.sel(
            time=storyline["time"].dt.year.isin(selected_years)
        )
        year_cross_med = year_cross_med.sel(
            time=year_cross_med["time"].dt.month.isin([6, 7, 8])
        )

        # Count months exceeding threshold
        exceedance_count = (year_cross_med > 294.26).sum(dim="time")

        # Subset data if lat/lon range is provided
        if lat_range and lon_range:
            exceedance_count = exceedance_count.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        lat = exceedance_count.lat.values
        lon = exceedance_count.lon.values

        # Step 1: Create 2D lat/lon grid
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Step 2: Get natural earth land regions (entire Regions object, not a single region)
        regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110

        mask = regions.mask(lon2d, lat2d)

        # Step 4: Convert to boolean mask: True = land, False = ocean
        land_mask = ~np.isnan(mask)

        # Step 4: Convert to xarray DataArray and align with exceedance_count
        land_mask_xr = xr.DataArray(
            land_mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )

        # Step 5: Apply mask correctly
        exceedance_count_masked = exceedance_count.where(land_mask_xr)

        # Define a discrete colormap for clear segmentation
        cmap = cm.get_cmap("plasma_r", vmax - vmin + 1)
        norm = mcolors.BoundaryNorm(np.arange(vmin, vmax + 2), cmap.N)

        # Plot on subplot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )

        ax.coastlines(linewidth=1.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="white",
                linewidth=0.7,
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            exceedance_count_masked,
            # masked_exceedance,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        ax.set_facecolor("white")  # White background = ocean
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="physical",
                name="land",
                scale="50m",
                facecolor="none",
                edgecolor="none",
            ),
            zorder=3,
        )

        # Add IPCC region mask
        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar setup with discrete bins
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])
    cbar = fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", extend="neither", pad=0.05
    )
    cbar.set_label("Months Exceeding Heat Stress Level", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"Storyline Analysis: Count of Months Exceeding Heat Stress Level",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    # plt.show()
    if save == True:
        plt.savefig(
            f"{config['figure_dir']}analysis/{config['expname']}_slines_danger_count_{temp_target}_{num_years}yravg_seed{seed}.png",
            dpi=300,
        )


def plot_raw_temp(
    storyline_matches,
    d_test,
    d_obs,
    config,
    num_years=1,
    lat_range=None,
    lon_range=None,
    vmin=280,
    vmax=300,
):
    """
    Plots temperature maps for multiple SSPs in a 4-panel subplot.

    Parameters:
        storyline_matches (list): List of storyline matches with SSP, GCM, and member info.
        data (object): Data containing test dataset.
        config (object): Configuration for computing anomalies.
        lat_range (tuple): Latitude range for subsetting.
        lon_range (tuple): Longitude range for subsetting.
        projection alternative options: ccrs.EqualEarth(central_longitude=250)
    """

    num_ssps = len(storyline_matches)  # Get the number of SSPs
    num_cols = min(2, num_ssps)  # Use up to 2 columns
    num_rows = math.ceil(num_ssps / num_cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()  # Flatten in case of a single row or column

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tasmin_Amon_hist_{ssp}_{model}_r1-10i1p1f1_monthly_gn_185001-210012_2pt5deg.nc"
        )

        # Compute anomalies
        # da_anomalies = compute_gridded_anomalies(s1, config)

        # Select storyline based on test member
        storyline = s1[member, :, :, :]

        # Isolate median year of threshold crossing prediction
        # year_cross_med = storyline.sel(
        #     time=storyline["time"].dt.year == (year_obs_prediction + year_test_value)
        # )

        year_cross_med = storyline.sel(
            time=storyline["time"].dt.year.isin(
                range(
                    int(year_obs_prediction + year_test_value),
                    int(year_obs_prediction + year_test_value) + num_years,
                )
            )
        )

        year_cross_med = year_cross_med.sel(
            time=year_cross_med["time"].dt.month.isin([6, 7, 8])
        )

        year_cross_med = year_cross_med.mean("time")

        # Subset data if lat/lon range is provided
        if lat_range and lon_range:
            year_cross_med = year_cross_med.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        # Get lat/lon values
        lat = year_cross_med.lat.values
        lon = year_cross_med.lon.values

        # Set up colormap
        # cmap = cmaps_ncl.BlueDarkRed18.colors
        # cmap = plt.cm.get_cmap("RdBu_r")  # Use a diverging colormap
        cmap = cm.get_cmap("Reds")

        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Create a normalizer for discrete bins
        # bounds = np.round(np.linspace(0, 5, 11), 2)  # Creates segments
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot on subplot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )
        # ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", edgecolor="k", facecolor="None"))
        # Add country and state borders
        ax.coastlines(linewidth=1.5)  # Thicker country borders
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)  # International borders
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,  # Thicker state borders
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            year_cross_med[:, :],
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add IPCC region mask
        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar setup
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", shrink=0.8, extend="both", pad=0.05
    )
    cbar.set_label("Temperature Anomaly (°C)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Adjust font size of tick labels

    # Main title
    fig.suptitle(
        f"Storyline Analysis after Crossing the {temp_target}°C Regional Threshold \n Minimum JJA Temperature {num_years}-year Average",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


# HAVEN'T IMPLEMENTED THESE FUNCITONS YET
def plot_storyline_single(
    xplot,
    lat,
    lon,
    ipcc_region,
    lat_range=None,
    lon_range=None,
    title=None,
    colorbar=True,
):
    # Subset data
    if lat_range or lon_range:
        xplot = xplot.sel(
            lat=slice(min(lat_range), max(lat_range)),
            lon=slice(min(lon_range), max(lon_range)),
        )

    # Update lat and lon after subsetting the data
    lat = xplot.lat.values
    lon = xplot.lon.values

    # c = cmaps_ncl.BlueDarkRed18.colors
    # cmap = mpl.colors.ListedColormap(c)
    cmap = cm.get_cmap("Reds")

    transform = ct.crs.PlateCarree()
    projection = ct.crs.EqualEarth(central_longitude=250)
    # Use LambertConformal projection, centered over North America
    # projection = ccrs.LambertConformal(
    #     central_longitude=-98.0,  # Central meridian for North America
    #     central_latitude=39.0,    # Central latitude for North America
    #     standard_parallels=(38, 40),  # Standard parallels for North America
    # )

    fig = plt.figure(figsize=(4, 3), dpi=200)
    a1 = fig.add_subplot(1, 1, 1, projection=projection)

    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", edgecolor="k", facecolor="None"
        )
    )
    c1 = a1.pcolormesh(
        lon,
        lat,
        xplot,
        cmap=cmap,
        transform=transform,
    )
    a1.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "110m",
            edgecolor="k",
            linewidth=0.5,
            facecolor="None",
        )
    )
    regionmask.defined_regions.ar6.land[(ipcc_region,)].plot(
        add_label=False,
        label_multipolygon="all",
        add_ocean=False,
        ocean_kws=dict(color="lightblue", alpha=0.25),
        line_kws=dict(
            linewidth=1.0,
        ),
    )

    c1.set_clim(0, 5)
    if colorbar:
        fig.colorbar(
            c1,
            orientation="horizontal",
            shrink=0.50,
            extend="both",
            pad=0.02,
            label="Change in Temperature (C)",
        )
    if title is not None:
        plt.title(title)


def xai_storylines_integrated_gradients(
    storyline_matches,
    d_test,
    d_obs,
    config,
    num_years=1,
    lat_range=None,
    lon_range=None,
    vmin=-3,
    vmax=3,
):
    """
    Plots temperature maps for multiple SSPs in a 4-panel subplot.

    Parameters:
        storyline_matches (list): List of storyline matches with SSP, GCM, and member info.
        data (object): Data containing test dataset.
        config (object): Configuration for computing anomalies.
        lat_range (tuple): Latitude range for subsetting.
        lon_range (tuple): Longitude range for subsetting.
        projection alternative options: ccrs.EqualEarth(central_longitude=250)
    """

    num_ssps = len(storyline_matches)  # Get the number of SSPs
    num_cols = min(2, num_ssps)  # Use up to 2 columns
    num_rows = math.ceil(num_ssps / num_cols)  # Compute number of rows dynamically

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()  # Flatten in case of a single row or column

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tasmin_Amon_hist_{ssp}_{model}_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc"
        )

        # Compute anomalies
        da_anomalies = compute_gridded_anomalies(s1, config)

        # Select storyline based on test member
        storyline = da_anomalies[member, :, :, :]

        # Isolate median year of threshold crossing prediction
        # year_cross_med = storyline.sel(
        #     time=storyline["time"].dt.year == (year_obs_prediction + year_test_value)
        # )

        year_cross_med = storyline.sel(
            time=storyline["time"].dt.year.isin(
                range(
                    int(year_obs_prediction + year_test_value),
                    int(year_obs_prediction + year_test_value) + num_years,
                )
            )
        )

        year_cross_med = year_cross_med.mean("time")

        # Subset data if lat/lon range is provided
        if lat_range and lon_range:
            year_cross_med = year_cross_med.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        # Get lat/lon values
        lat = year_cross_med.lat.values
        lon = year_cross_med.lon.values

        # Set up colormap
        # cmap = cmaps_ncl.BlueDarkRed18.colors
        # cmap = plt.cm.get_cmap("RdBu_r")  # Use a diverging colormap
        cmap = cm.get_cmap("Reds")

        # norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Create a normalizer for discrete bins
        bounds = np.round(np.linspace(0, 5, 11), 2)  # Creates segments
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot on subplot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )
        # ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", edgecolor="k", facecolor="None"))
        # Add country and state borders
        ax.coastlines(linewidth=1.5)  # Thicker country borders
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)  # International borders
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,  # Thicker state borders
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            year_cross_med[:, :],
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add IPCC region mask
        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar setup
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", shrink=0.8, extend="both", pad=0.05
    )
    cbar.set_label("Temperature Anomaly (°C)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)  # Adjust font size of tick labels

    # Main title
    fig.suptitle(
        f"Storyline Analysis after Crossing the {temp_target}°C Regional Threshold \n Minimum JJA Temperature {num_years}-year Average",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()


def plot_raw_temp_exceedance_change_window(
    storyline_matches,
    d_test,
    d_obs,
    config,
    seed,
    lat_range=None,
    lon_range=None,
    vmin=0,
    vmax=10,  # For difference
    save=True,
):
    """
    Plots maps showing the difference in the number of JJA months exceeding a temperature threshold
    between the 15 years after and 15 years before a crossing year.

    Danger Days Change = (# after crossing) - (# before crossing)
    """

    num_ssps = len(storyline_matches)
    num_cols = min(2, num_ssps)
    num_rows = math.ceil(num_ssps / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()

    for i, match in enumerate(storyline_matches):
        ssp = match["ssp"]
        model = match["gcm_name"]
        temp_target = match["temp_target"]
        member = match["member"]
        year_obs_prediction = match["year"]
        year_test_value = match["test_value"]
        year_obs_value = match["obs_value"]
        year_cross = int(year_obs_prediction + year_test_value)

        # Load data
        ddir_data = "/Users/marcodia/Documents/Projects/temp_thresh_shash/data/"
        s1 = xr.open_dataarray(
            f"{ddir_data}tasmin_Amon_hist_{ssp}_{model}_r1-10i1p1f1_monthly_gn_185001-210012_2pt5deg.nc"
        )

        storyline = s1[member, :, :, :]

        # Select years before and after crossing
        years_before = range(year_cross - 15, year_cross)
        years_after = range(year_cross, year_cross + 15)

        def count_exceedance(years):
            subset = storyline.sel(time=storyline.time.dt.year.isin(years))
            subset = subset.sel(time=subset.time.dt.month.isin([6, 7, 8]))  # JJA
            count = (subset > 294.26).sum(dim="time")
            return count

        count_before = count_exceedance(years_before)
        count_after = count_exceedance(years_after)
        exceedance_diff = count_after - count_before

        if lat_range and lon_range:
            exceedance_diff = exceedance_diff.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        lat = exceedance_diff.lat.values
        lon = exceedance_diff.lon.values

        # Step 1: Create 2D lat/lon grid
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Step 2: Get natural earth land regions (entire Regions object, not a single region)
        regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110

        mask = regions.mask(lon2d, lat2d)

        # Step 4: Convert to boolean mask: True = land, False = ocean
        land_mask = ~np.isnan(mask)

        # Step 4: Convert to xarray DataArray and align with exceedance_count
        land_mask_xr = xr.DataArray(
            land_mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )

        # Step 5: Apply mask correctly
        exceedance_diff = exceedance_diff.where(land_mask_xr)

        # Define range and boundaries
        bounds = np.arange(vmin, vmax + 2)  # +2 to include upper bound
        n_colors = len(bounds) - 1
        cmap = cm.get_cmap("hot_r", n_colors)  # discrete colormap with n_colors bins
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
        # Plot
        ax = axes[i]
        ax.set_title(
            f"{ssp} - {model} ens #{member} crosses in {np.round(year_obs_prediction + year_test_value).astype(int)} \n (Obs prediction: {np.round(year_obs_prediction + year_obs_value).astype(int)})",
            fontsize=14,
        )
        ax.coastlines(linewidth=1.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="white",
                linewidth=0.7,
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            exceedance_diff,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )
        ax.set_facecolor("white")  # White background = ocean
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="physical",
                name="land",
                scale="50m",
                facecolor="none",
                edgecolor="none",
            ),
            zorder=3,
        )

        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])
    cbar = fig.colorbar(
        c,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=bounds,
        spacing="uniform",
        pad=0.05,
        extend="both",
    )
    cbar.set_label("Months Exceeding Heat Stress Level", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"Storyline Analysis: Increased Number of Months Exceeding Heat Stress Level",  # \nafter Crossing the {temp_target}°C Regional Threshold",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()


def plot_avg_temp_after_threshold_by_ssp(
    data_all,
    config,
    num_years=1,
    temp_target=None,
    lat_range=None,
    lon_range=None,
    vmin=-3,
    vmax=3,
    save=True,
):
    """
    Plots gridded average temperature anomalies for each SSP after the threshold year.

    Parameters:
        data_all (dict or list): Dict of arrays (e.g., 'ssp', 'gcm_name', ...) or list of dicts.
        config (dict): Configuration for anomaly computation and paths.
        num_years (int): Number of years to average after year_reached.
        lat_range (tuple): Optional lat range (min, max).
        lon_range (tuple): Optional lon range (min, max).
    """

    # Convert dict of arrays to list of dicts, if needed
    if isinstance(data_all, dict):
        keys = data_all.keys()
        num_entries = len(data_all[next(iter(keys))])
        data_all = [{k: data_all[k][i] for k in keys} for i in range(num_entries)]

    # Collapse to one entry per simulation using year_reached
    grouped_years = {}
    for entry in data_all:
        ssp = entry["ssp"]
        tt = entry["temp_target"]
        gcm = entry["gcm_name"]
        member = entry["member"]
        year = entry["year_reached"]

        if temp_target is None or tt == temp_target:
            key = (ssp, tt, gcm, member)
            grouped_years[key] = year

    # Group by SSP
    ssp_groups = defaultdict(list)
    for (ssp, tt, gcm, member), year in grouped_years.items():
        ssp_groups[ssp].append(
            {"temp_target": tt, "gcm": gcm, "member": member, "year_reached": year}
        )

    num_ssps = len(ssp_groups)
    num_cols = min(2, num_ssps)
    num_rows = math.ceil(num_ssps / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()

    cmap = cm.get_cmap("Reds")
    bounds = np.round(np.linspace(vmin, vmax, vmax * 4 + 1), 8)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i, (ssp, entries) in enumerate(ssp_groups.items()):
        anomaly_list = []

        for entry in entries:
            gcm = entry["gcm"]
            member = entry["member"]
            year_reached = entry["year_reached"]
            # Skip all ensemble members who don't reach threshold
            if year_reached >= 2086.0:
                print(
                    f"Skipping {gcm} member {member} in {ssp} due to year_reached={year_reached}"
                )
                continue

            file_path = f"{config['data_dir']}tasmin_Amon_hist_{ssp}_{gcm}_r1-10i1p1f1_JJA_gn_185001-210012_2pt5deg.nc"
            try:
                da = xr.open_dataarray(file_path)
            except FileNotFoundError:
                print(f"Missing: {file_path}")
                continue

            # Compute anomalies
            da_anom = compute_gridded_anomalies(da, config)

            if member >= da_anom.shape[0]:
                print(f"Invalid member {member} for {gcm} in {ssp}")
                continue

            # Get the specific member
            member_data = da_anom[member, :, :, :]

            # Select years and compute average
            selected_years = member_data.sel(
                time=member_data.time.dt.year.isin(
                    range(int(year_reached), int(year_reached + num_years))
                )
            )
            # print(
            #     f"Selected years for {gcm} {member}: {selected_years.time[0]}, {selected_years.time[-1]}"
            # )

            avg_anomaly = selected_years.mean("time")
            anomaly_list.append(avg_anomaly)

        if not anomaly_list:
            print(f"No valid data for {ssp}")
            continue

        # Stack/average all anomalies across models
        print(f"Number of anomalies for {ssp}: {len(anomaly_list)}")
        ssp_avg_anomaly = xr.concat(anomaly_list, dim="ensemble").mean("ensemble")

        # Subset if needed
        if lat_range and lon_range:
            ssp_avg_anomaly = ssp_avg_anomaly.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        lat = ssp_avg_anomaly.lat.values
        lon = ssp_avg_anomaly.lon.values

        ax = axes[i]
        ax.set_title(f"{ssp} (n={len(anomaly_list)})", fontsize=14)
        ax.coastlines(linewidth=1.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            ssp_avg_anomaly,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])
    fig.colorbar(
        c, cax=cbar_ax, orientation="horizontal", shrink=0.8, extend="both", pad=0.05
    ).set_label("Temperature Anomaly (°C)", fontsize=14)

    fig.suptitle(
        f"Min JJA Temp Anomaly after {num_years}-year threshold (per SSP)",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    if save:
        plt.savefig(
            f"{config['figure_dir']}analysis/{config['expname']}_avg_after_thresh_{num_years}yrs.png",
            dpi=300,
        )


def plot_avg_raw_temp_exceedance_change_window_by_ssp(
    data_all,
    config,
    temp_target,
    lat_range=None,
    lon_range=None,
    vmin=0,
    vmax=10,  # For difference
    save=True,
):
    """
    Plots FORCED RESPONSE maps showing the difference in the number of JJA months exceeding a temperature threshold
    between the 15 years after and 15 years before a crossing year.

    Danger Days Change = (# after crossing) - (# before crossing)
    """

    # Convert dict of arrays to list of dicts, if needed
    if isinstance(data_all, dict):
        keys = data_all.keys()
        num_entries = len(data_all[next(iter(keys))])
        data_all = [{k: data_all[k][i] for k in keys} for i in range(num_entries)]

    # Collapse to one entry per simulation using year_reached
    grouped_years = {}
    for entry in data_all:
        ssp = entry["ssp"]
        tt = entry["temp_target"]
        gcm = entry["gcm_name"]
        member = entry["member"]
        year = entry["year_reached"]

        if temp_target is None or tt == temp_target:
            key = (ssp, tt, gcm, member)
            grouped_years[key] = year

    # Group by SSP
    ssp_groups = defaultdict(list)
    for (ssp, tt, gcm, member), year in grouped_years.items():
        ssp_groups[ssp].append(
            {"temp_target": tt, "gcm": gcm, "member": member, "year_reached": year}
        )

    num_ssps = len(ssp_groups)
    num_cols = min(2, num_ssps)
    num_rows = math.ceil(num_ssps / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10, 4 * num_rows),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.array(axes).flatten()

    for i, (ssp, entries) in enumerate(ssp_groups.items()):
        exceedance_diff_all = []

        for entry in entries:
            gcm = entry["gcm"]
            member = entry["member"]
            year_reached = entry["year_reached"]
            year_reached = int(year_reached)
            # Skip all ensemble members who don't reach threshold
            if year_reached >= 2086.0:
                print(
                    f"Skipping {gcm} member {member} in {ssp} due to year_reached={year_reached}"
                )
                continue

            file_path = f"{config['data_dir']}tasmin_Amon_hist_{ssp}_{gcm}_r1-10i1p1f1_monthly_gn_185001-210012_2pt5deg.nc"
            da = xr.open_dataarray(file_path)
            storyline = da[member, :, :, :]

            # Select years before and after crossing
            years_before = range(year_reached - 15, year_reached)
            years_after = range(year_reached, year_reached + 15)

            def count_exceedance(years):
                subset = storyline.sel(time=storyline.time.dt.year.isin(years))
                subset = subset.sel(time=subset.time.dt.month.isin([6, 7, 8]))  # JJA
                count = (subset > 294.26).sum(dim="time")
                return count

            count_before = count_exceedance(years_before)
            count_after = count_exceedance(years_after)
            exceedance_diff_single = count_after - count_before

            exceedance_diff_all.append(exceedance_diff_single)

        exceedance_diff = xr.concat(exceedance_diff_all, dim="ensemble").mean(
            "ensemble"
        )

        if lat_range and lon_range:
            exceedance_diff = exceedance_diff.sel(
                lat=slice(min(lat_range), max(lat_range)),
                lon=slice(min(lon_range), max(lon_range)),
            )

        lat = exceedance_diff.lat.values
        lon = exceedance_diff.lon.values

        # Step 1: Create 2D lat/lon grid
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Step 2: Get natural earth land regions (entire Regions object, not a single region)
        regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110

        mask = regions.mask(lon2d, lat2d)

        # Step 4: Convert to boolean mask: True = land, False = ocean
        land_mask = ~np.isnan(mask)

        # Step 4: Convert to xarray DataArray and align with exceedance_count
        land_mask_xr = xr.DataArray(
            land_mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )

        # Step 5: Apply mask correctly
        exceedance_diff = exceedance_diff.where(land_mask_xr)

        # Define range and boundaries
        bounds = np.arange(vmin, vmax + 2)  # +2 to include upper bound
        n_colors = len(bounds) - 1
        cmap = cm.get_cmap("hot_r", n_colors)  # discrete colormap with n_colors bins
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
        # Plot
        ax = axes[i]
        ax.set_title(f"{ssp} (n={len(exceedance_diff_all)})", fontsize=14)
        ax.coastlines(linewidth=1.5)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces",
                scale="50m",
                facecolor="none",
                edgecolor="white",
                linewidth=0.7,
            )
        )

        c = ax.pcolormesh(
            lon,
            lat,
            exceedance_diff,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        ax.set_facecolor("white")  # White background = ocean
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                category="physical",
                name="land",
                scale="50m",
                facecolor="none",
                edgecolor="none",
            ),
            zorder=3,
        )

        regionmask.defined_regions.ar6.land[("CNA",)].plot(
            add_label=False,
            label_multipolygon="all",
            add_ocean=False,
            ocean_kws=dict(color="lightblue", alpha=0.25),
            line_kws=dict(linewidth=4.0, edgecolor="black"),
            ax=ax,
        )

    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.02])
    cbar = fig.colorbar(
        c,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=bounds,
        spacing="uniform",
        pad=0.05,
        extend="both",
    )
    cbar.set_label("Months Exceeding Heat Stress Level", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"Storyline Analysis: Increased Number of Months Exceeding Heat Stress Level",  # \nafter Crossing the {temp_target}°C Regional Threshold",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

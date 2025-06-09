"""Data maker modules.

Functions
---------


Classes
---------
ClimateData_Transfer()
This class is updated from ClimateData() to scale the observation data to the climate model data for transfer learning.

ClimateData() is designed to handle climate data, including fetching, processing, and organizing it into different datasets
(e.g., training, validation, and test sets).
It also supports operations of selecting specific members from the data, computing anomalies, and calibrating observations.

"""

import matplotlib
import os

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import numpy as np

import utils
import visuals.plots as plots
import datamaker.emissions as emissions
import datamaker.regions as regions
import datamaker.filemethods as filemethods
from datamaker.sample_vault import SampleDict
import xarray


class ClimateDataTransfer:
    """
    Custom dataset for climate data and processing.
    """

    def __init__(self, config, expname, seed, data_dir, figure_dir, verbose=False):

        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose

        self.fetch_data()
        self.fetch_obs()

    def fetch_obs(
        self,
        # ssp,
        # temp_target_list=None,
        verbose=None,
        # output_obs_dataset=None,
        plot=True,
        #    include_all=False,  # Additional parameter to control data handling
    ):
        if verbose is not None:
            self.verbose = verbose

        self.d_obs = SampleDict()  # store the obs

        self._create_obs()

        if self.verbose:
            print("-----------------")
            self.d_obs.summary()

    def _create_obs(
        self,
        plot=True,
        #    include_all=False,
    ):
        for ssp in self.config["ssp_list"]:
            self._ssp = ssp

            # Load input and target observational data
            # I don't need to specify input/output variable and period as there is only 1 file to read in for each obs file

            input_filename = filemethods.get_observations_filename_input(
                source=self.config["obs_source"], verbose=self.verbose
            )
            # print(f"input_filename: {input_filename}")
            da = filemethods.get_netcdf_da(self.data_dir + input_filename)
            da = da.expand_dims(dim="member", axis=0)

            # if output_obs_dataset is None:
            output_filename = filemethods.get_observations_filename_output(
                source=self.config["output_obs_source"], verbose=self.verbose
            )
            # print(f"output_filename: {output_filename}")
            # else:
            #     output_filename = output_obs_dataset

            da_y = filemethods.get_netcdf_da(self.data_dir + output_filename)
            da_y = da_y.expand_dims(dim="member", axis=0)

            # for temp_target in self.config["obs_temp_targets"]:
            for temp_target in self.config["temp_targets"]:
                f_dict = self._process_data(
                    da,
                    da_y,
                    temp_target,
                    members=(0,),  # Assuming single member for observations
                    gcm_name=self.config["obs_source"],
                )
                f_dict = self._select_training_years(f_dict)

                # Skip data split if include_all is True
                # if include_all:
                #     f_dict = self._select_training_years(f_dict, include_all=True)
                # else:
                #     f_dict = self._select_training_years(f_dict)

                # Concatenate with the rest of the observational data
                # print("Before concatenation OBS:")
                # print(f"f_dict['x'].shape: {f_dict['x'].shape}")
                self.d_obs.concat(f_dict)
                # print("After concatenation:")
                # print(f"self.d_obs['x'].shape: {self.d_obs['x'].shape}")

        # print("Before reshape:")
        # print(f"self.d_obs['x'].shape: {self.d_obs['x'].shape}")
        # Reshape the data into samples
        self.d_obs.reshape()
        # print("After reshape:")
        # print(f"self.d_obs['x'].shape: {self.d_obs['x'].shape}")

        # Apply filtering if required
        if self.config["filter_historical"]:
            self.d_obs = self._apply_filtering(self.d_obs)

        # Fill nans with zeros
        if self.config["anomalies"]:
            self.d_obs["x"] = np.nan_to_num(self.d_obs["x"], 0.0)

        # Add latitude and longitude
        self.lat = da.lat.values
        self.lon = da.lon.values

        # Cleanup
        del self._ssp

    def fetch_data(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose

        self.d_train = SampleDict()
        self.d_val = SampleDict()
        self.d_test = SampleDict()

        self._get_members()  # determine which members (subsets of data) will be used
        # self._create_input_data()
        self._create_data()  # process the data and split it into training, validation, and testing datasets

        if self.verbose:
            print("-----------------")
            self.d_train.summary()
            self.d_val.summary()
            self.d_test.summary()

    def _create_data(self):

        # get the SSP list
        for isub, sub in enumerate(self.config["gcmsub"]):
            # print(sub)
            for ssp in self.config["ssp_list"]:
                # print(ssp)
                self._ssp = ssp
                input_var = self.config["input_var"][0]
                self._input_var = input_var
                # print(input_var)
                input_period = self.config["input_period"][0]
                self._input_period = input_period
                # print(input_period)
                target_var = self.config["target_var"][0]
                self._target_var = target_var
                # print(target_var)
                target_period = self.config["target_period"][0]
                self._target_period = target_period
                # print(target_period)
                filenames = filemethods.get_cmip_filenames(
                    ssp, input_var, input_period, sub
                )
                filenames_y = filemethods.get_cmip_filenames(
                    ssp, target_var, target_period, sub
                )
                # print('filenames_X')
                # print(filenames)
                # print('filenames_Y')
                # print(filenames_y)

                for f, fy in zip(filenames, filenames_y):
                    # if self.verbose:
                    #     print(f)
                    #     print(fy)
                    da = filemethods.get_netcdf_da(self.data_dir + f)
                    da_y = filemethods.get_netcdf_da(self.data_dir + fy)
                    gcm_name = filemethods.get_gcm_name(f, ssp)

                    for temp_target in self.config["temp_targets"]:

                        # get processed X and Y data
                        # process the data, i.e. compute anomalies, subtract the mean, etc.
                        f_dict_train = self._process_data(
                            da,
                            da_y,
                            temp_target,
                            members=self.train_members[isub],
                            gcm_name=gcm_name,
                        )
                        f_dict_val = self._process_data(
                            da,
                            da_y,
                            temp_target,
                            members=self.val_members[isub],
                            gcm_name=gcm_name,
                        )
                        f_dict_test = self._process_data(
                            da,
                            da_y,
                            temp_target,
                            members=self.test_members[isub],
                            gcm_name=gcm_name,
                        )

                        # select training years
                        f_dict_train = self._select_training_years(f_dict_train)
                        f_dict_val = self._select_training_years(f_dict_val)
                        f_dict_test = self._select_training_years(f_dict_test)

                        # # concatenate with the rest of the data

                        # print("Before concatenation:")
                        # print(f"f_dict_train['x'].shape: {f_dict_train['x'].shape}")
                        # print(f"f_dict_val['x'].shape: {f_dict_val['x'].shape}")
                        # print(f"f_dict_test['x'].shape: {f_dict_test['x'].shape}")

                        self.d_train.concat(f_dict_train)
                        self.d_val.concat(f_dict_val)
                        self.d_test.concat(f_dict_test)

                        # print("After concatenation:")
                        # print(f"self.d_train['x'].shape: {self.d_train['x'].shape}")
                        # print(f"self.d_val['x'].shape: {self.d_val['x'].shape}")
                        # print(f"self.d_test['x'].shape: {self.d_test['x'].shape}")

        # reshape the data into samples
        print("Before reshape:")
        print(f"self.d_train['x'].shape: {self.d_train['x'].shape}")
        self.d_train.reshape()
        self.d_val.reshape()
        self.d_test.reshape()

        print("After reshape:")
        print(f"self.d_train['x'].shape: {self.d_train['x'].shape}")

        # Apply filtering for repetitive data after reshaping
        if self.config["filter_historical"] is True:
            self.d_train = self._apply_filtering(self.d_train)
            self.d_val = self._apply_filtering(self.d_val)
            self.d_test = self._apply_filtering(self.d_test)

        # clean-up the train/val/test data to remove np.nan samples
        self.d_train.del_nans()
        self.d_val.del_nans()
        self.d_test.del_nans()

        # add latitude and longitude
        self.lat = da.lat.values
        self.lon = da.lon.values

        # cleanup
        del self._ssp

    def _process_data(
        self,
        da,
        da_y,
        temp_target,
        members=None,
        gcm_name=None,
        verbose=None,
        plot=True,
    ):
        if verbose is not None:
            self.verbose = verbose

        # --------------
        # CREATE THE FILE-DATA DICTIONARY
        f_dict = SampleDict()

        # --------------
        # MEMBER ANALYSIS
        # check if any members will be grabbed at all
        if len(members) == 0:
            return f_dict

        # select the members
        if members is None:
            da_members = da
        else:
            if not isinstance(members, np.ndarray):
                members = np.asarray(members)
            da_members = da[members, :, :, :]

        # --------------
        # ADD CHANNEL DIMENSION
        da_members = da_members.expand_dims(dim={"channel": 1}, axis=-1)

        # --------------
        # GET THE SCALAR DATA (e.g. labels, years, etc.)
        f_dict = self._get_scalar_data(
            f_dict, da_y, temp_target, members, figname=gcm_name, plot=True
        )

        # --------------
        # GET THE X-DATA
        f_dict = self._get_gridded_data(f_dict, da_members)

        # --------------
        # GET EMISSIONS
        _, cum_emiss_left, _ = emissions.create_emissions(
            self._ssp, f_dict["year"][0, :], self.data_dir, self.figure_dir, plot=True
        )
        f_dict["emissions_left"] = np.tile(cum_emiss_left, (f_dict["y"].shape[0], 1))
        if self.config["use_emissions_input"] is False:
            f_dict["emissions_left"] = f_dict["emissions_left"] * 0.0

        # --------------
        # INSERT META DATA
        f_dict["gcm_name"] = np.tile(gcm_name, f_dict["y"].shape)
        f_dict["ssp"] = np.tile(self._ssp, f_dict["y"].shape)
        f_dict["member"] = np.tile(members[:, np.newaxis], f_dict["y"].shape[1])

        return f_dict

    def _get_gridded_data(self, f_dict, da):
        if self.config["anomalies"] is True:
            da_anomalies = da - da.sel(
                time=slice(
                    str(self.config["anomaly_yr_bounds"][0]),
                    str(self.config["anomaly_yr_bounds"][1]),
                )
            ).mean("time")
        elif self.config["anomalies"] == "baseline":
            # if self.verbose:
            #     print("computing anomalies relative to baseline...")
            da_anomalies = da - da.sel(
                time=slice(
                    str(self.config["baseline_yr_bounds"][0]),
                    str(self.config["baseline_yr_bounds"][1]),
                )
            ).mean("time")
            # print(da_anomalies)
            # print("computing anomalies relative to recent reference period")
            da_anomalies = da_anomalies - da_anomalies.sel(
                time=slice(
                    str(self.config["anomaly_yr_bounds"][0]),
                    str(self.config["anomaly_yr_bounds"][1]),
                )
            ).mean("time")
            # print(da_anomalies)

        elif not self.config["anomalies"]:
            print("not computing any anomalies...")
            pass
        else:
            raise NotImplementedError()

        if self.config["remove_map_mean"] == "raw":
            da_anomalies = da_anomalies - da_anomalies.mean(("lon", "lat"))
        elif self.config["remove_map_mean"] == "weighted":
            weights = np.cos(np.deg2rad(da_anomalies.lat))
            weights.name = "weights"
            da_anomalies_weighted = da_anomalies.weighted(weights)
            da_anomalies = da_anomalies - da_anomalies_weighted.mean(("lon", "lat"))

        f_dict["x"] = da_anomalies.values

        return f_dict

    def _get_scalar_data(
        self, f_dict, da_y, temp_target, members, figname="", plot=False
    ):
        # if verbose is not None:
        #     self.verbose = verbose

        data_output = regions.extract_region(
            da_y,
            region=self.config["target_region"],
            dir=self.config["data_dir"],
            lat=da_y.lat,
            lon=da_y.lon,
        )
        global_mean = regions.compute_global_mean(data_output.mean(axis=0))
        # output: (time); global mean temperature for each year averaged across all ensemble members
        global_mean_ens = regions.compute_global_mean(data_output)
        # output: (member,time); one mean temperature time series per ensemble member

        # compute the baseline mean for scaling all ensemble members to the same zero
        baseline_mean = global_mean.sel(
            time=slice(
                str(self.config["baseline_yr_bounds"][0]),
                str(self.config["baseline_yr_bounds"][1]),
            )
        ).mean("time")
        # output: scalar value of single baseline temperature value for baseline_yr_bounds

        # grab each member separately
        years = global_mean_ens[
            "time.year"
        ].values  # array of years for dictionary data
        anomalies = (
            global_mean_ens - baseline_mean
        ).values  # output: (member,time); computes temperature anomalies by subtracting baseline_mean from each ensemble member
        anomalies_ens = global_mean_ens - baseline_mean
        anomalies_mean = anomalies.mean(
            axis=0
        )  # output: (time); computes mean temperature anomaly across all ensemble members

        # Read parameters from config
        threshold = self.config.get("threshold", "no_return")
        thresh_time = self.config.get("thresh_time", None)

        if threshold == "prolonged_exceedance" and thresh_time is None:
            raise ValueError(
                "`thresh_time` must be specified in the config for 'prolonged_exceedance'."
            )

        # compute when the temp_target is reached
        temp_target = np.ones((anomalies.shape[0])) * temp_target
        year_reached = np.zeros((anomalies.shape[0],))

        # this is making the "no-return" threshold- crosses thresh but never goes below again
        # No-return threshold
        if threshold == "no_return":
            for mem in range(anomalies.shape[0]):
                i_lastyear = np.where((anomalies[mem, ...] - temp_target[mem]) < 0)[0]
                if len(i_lastyear) < 1:
                    year_reached[mem] = 2100
                elif i_lastyear[-1] == len(years) - 1:
                    year_reached[mem] = 2100
                else:
                    year_reached[mem] = years[i_lastyear[-1] + 1]

        # Prolonged exceedance threshold
        elif threshold == "prolonged_exceedance":
            for mem in range(anomalies.shape[0]):
                # Check for `thresh_time` consecutive years exceeding the target
                exceedances = (anomalies[mem, ...] > temp_target[mem]).astype(int)
                # Find indices where `thresh_time` consecutive years exceed the target
                for i in range(len(exceedances) - thresh_time + 1):
                    if np.all(
                        exceedances[i : i + thresh_time] == 1
                    ):  # Check for consecutive years
                        year_reached[mem] = years[i]
                        # year_reached[mem] = years[
                        #     i + thresh_time - 1
                        # ]  # Final year of the period
                        break
                else:
                    year_reached[mem] = 2100  # Default if no such period is found

        # --------------------
        # plot the results
        if plot:
            # plot the results
            plots.plot_anomaly_definition(
                year_reached,
                temp_target,
                anomalies_ens,
                anomalies_mean,
                anomalies,
                range(len(years)),
                self.config,
            )

            plt.title(
                figname + ": " + self._ssp + ", " + str(temp_target[0]), fontsize=8
            )
            plt.savefig(
                self.figure_dir
                + "data_diagnostics/"
                + self.expname
                + "_"
                + figname
                + "_"
                + self._ssp
                + "_"
                + str(temp_target[0])
                + "_anomaly_definition"
                + ".png",
                dpi=self.config["fig_dpi"],
            )
            plt.close()

            # plot the results
            plots.plot_single_anomaly_definition(
                year_reached,
                temp_target,
                anomalies_ens,
                anomalies,
                range(len(years)),
                self.config,
            )

            plt.title(
                figname + ": " + self._ssp + ", " + str(temp_target[0]), fontsize=8
            )
            plt.savefig(
                self.figure_dir
                + "data_diagnostics/"
                + self.expname
                + "_"
                + figname
                + "_"
                + self._ssp
                + "_"
                + str(temp_target[0])
                + "_single_anomaly_definition"
                + ".png",
                dpi=self.config["fig_dpi"],
            )
            plt.close()

        # --------------------
        # define the labels
        current_temp = global_mean_ens.values - baseline_mean.values
        labels = year_reached[:, np.newaxis] - years

        # ----------------------------------------------------------
        # ONLY GRAB THE MEMBERS THAT WE WANT
        if members is not None:
            labels = labels[members, :]
            current_temp = current_temp[members, :]
            temp_target = temp_target[members]
            year_reached = year_reached[members]

        assert labels.shape == current_temp.shape
        assert labels.shape[0] == temp_target.shape[0]

        if self.verbose:
            print(f"  {temp_target[0] = :2.2f}\n" f"  {year_reached = }")

        # ----------------------------------------------------------
        # RESHAPE DATA INTO SAMPLES AND STORE IN RETURNED DICTIONARY
        f_dict["y"] = labels
        f_dict["current_temp"] = current_temp
        f_dict["temp_target"] = np.tile(
            temp_target[:, np.newaxis], f_dict["y"].shape[1]
        )
        f_dict["year_reached"] = np.tile(
            year_reached[:, np.newaxis], f_dict["y"].shape[1]
        )
        print("YEAR REACHED: ", year_reached)
        f_dict["year"] = np.tile(da_y["time.year"].values, (f_dict["y"].shape[0], 1))
        assert np.sum(np.diff(f_dict["temp_target"], axis=1)) == 0

        # if self.verbose:
        #     print(f"  {f_dict['y'].shape = }")

        return f_dict

    def _get_members(self):

        self.train_members = []
        self.val_members = []
        self.test_members = []

        for splitvec in self.config["n_train_val_test"]:
            n_train = splitvec[0]
            n_val = splitvec[1]
            n_test = splitvec[2]
            all_members = np.arange(0, n_train + n_val + n_test)

            rng_cmip = np.random.default_rng(self.seed)
            train_members = rng_cmip.choice(all_members, size=n_train, replace=False)
            val_members = rng_cmip.choice(
                np.setdiff1d(all_members, train_members), size=n_val, replace=False
            )
            test_members = rng_cmip.choice(
                np.setdiff1d(all_members, np.append(train_members[:], val_members)),
                size=n_test,
                replace=False,
            )
            self.train_members.append(train_members.tolist())
            self.val_members.append(val_members.tolist())
            self.test_members.append(test_members.tolist())

            if self.verbose:
                print(
                    f"Member for train/val/test split: {self.train_members} / {self.val_members} / {self.test_members}"
                )

    def _select_training_years(self, f_dict):

        if len(f_dict["y"]) == 0:
            return f_dict

        # only train on certain samples
        iyears = copy.deepcopy(
            np.where(
                (f_dict["year"][0, :] >= self.config["training_yr_bounds"][0])
                & (f_dict["year"][0, :] <= self.config["training_yr_bounds"][1])
            )[0]
        )
        f_dict.subsample(idx=iyears, axis=1)
        return f_dict

    def _apply_filtering(self, f_dict):
        # Check if the dataset is empty
        if len(f_dict["y"]) == 0:
            return f_dict

        # Find indices of SSPs that are not 'ssp245'
        ssp_mask = f_dict["ssp"] != "ssp245"

        # After reshaping, the first dimension is the combination of ensemble and years.
        # Extract the reshaped 'year' and 'ssp' arrays and apply filtering based on non_historical_year bounds.

        # Identify indices of the non-historical years for SSPs other than 'ssp245'
        non_historical_years = np.where(
            (f_dict["year"] >= self.config["non_historical_yr_bounds"][0])
            & (f_dict["year"] <= self.config["non_historical_yr_bounds"][1])
        )[0]

        # Create a mask to filter out invalid years only for SSPs that are not 'ssp245'
        filter_mask = np.ones(
            f_dict["year"].shape[0], dtype=bool
        )  # Start with all True
        filter_mask[ssp_mask] = np.isin(
            np.arange(f_dict["year"].shape[0])[ssp_mask], non_historical_years
        )

        # Apply the filter to all relevant fields
        for key in f_dict:
            if len(f_dict[key]) > 0:
                f_dict[key] = f_dict[key][filter_mask]

        return f_dict

#!/usr/bin/env python

import logging
import os

logging.captureWarnings(True)

import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.data.sources import (
    get_era5_reanalysis_data,
    get_earthenv_auxiliary_data,
    get_gldas_land_mask,
)

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm




# Training/data config
data_range = ("2010-01-01", "2019-12-31")
train_range = ("2010-01-01", "2018-12-31")
val_range = ("2019-01-01", "2019-12-31")
date_subsample_factor = 2
extent = "north_america"
era5_var_IDs = ["2m_temperature"]
lowres_auxiliary_var_IDs = ["elevation"]
cache_dir = "../../.datacache"
deepsensor_folder = "../deepsensor_config/"
verbose_download = True




era5_raw_ds = get_era5_reanalysis_data(
    era5_var_IDs,
    extent,
    date_range=data_range,
    cache=True,
    cache_dir=cache_dir,
    verbose=verbose_download,
    num_processes=8,
)
lowres_aux_raw_ds = get_earthenv_auxiliary_data(
    lowres_auxiliary_var_IDs,
    extent,
    "100KM",
    cache=True,
    cache_dir=cache_dir,
    verbose=verbose_download,
)
land_mask_raw_ds = get_gldas_land_mask(
    extent, cache=True, cache_dir=cache_dir, verbose=verbose_download
)

data_processor = DataProcessor(x1_name="lat", x2_name="lon")
era5_ds = data_processor(era5_raw_ds)
lowres_aux_ds, land_mask_ds = data_processor(
    [lowres_aux_raw_ds, land_mask_raw_ds], method="min_max"
)

dates = pd.date_range(era5_ds.time.values.min(), era5_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")
lowres_aux_ds["cos_D"] = doy_ds["cos_D"]
lowres_aux_ds["sin_D"] = doy_ds["sin_D"]




set_gpu_default_device()


# ## Initialise TaskLoader and ConvNP model



task_loader = TaskLoader(
    context=[era5_ds, land_mask_ds, lowres_aux_ds],
    target=era5_ds,
)
task_loader.load_dask()
print(task_loader)




# Set up model
model = ConvNP(data_processor, task_loader, unet_channels=(32, 32, 32, 32, 32))


# ## Define how Tasks are generated
# 

def gen_training_tasks(dates, progress=True):
    tasks = []
    for date in tqdm(dates, disable=not progress):
        tasks_per_date = task_loader(
            date,
            context_sampling=["all", "all", "all"],
            target_sampling="all",
            patch_strategy="random",
            patch_size=(0.4, 0.4),
            num_samples_per_date=2,
        )
        tasks.extend(tasks_per_date)
    return tasks


def gen_validation_tasks(dates, progress=True):
    tasks = []
    for date in tqdm(dates, disable=not progress):
        tasks_per_date = task_loader(
            date,
            context_sampling=["all", "all", "all"],
            target_sampling="all",
            patch_strategy="sliding",
            patch_size=(0.5, 0.5),
            stride=(1,1)
        )
        tasks.extend(tasks_per_date)
    return tasks


# ## Generate validation tasks for testing generalisation



val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]
val_tasks = gen_validation_tasks(val_dates)


# ## Training with the Trainer class




def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))




num_epochs = 50
losses = []
val_rmses = []

# # Train model
val_rmse_best = np.inf
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(num_epochs)):
    train_tasks = gen_training_tasks(pd.date_range(train_range[0], train_range[1])[::date_subsample_factor], progress=False)
    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))
    val_rmses.append(compute_val_rmse(model, val_tasks))
    if val_rmses[-1] < val_rmse_best:
        val_rmse_best = val_rmses[-1]
        model.save(deepsensor_folder)




fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses)
axes[1].plot(val_rmses)
_ = axes[0].set_xlabel("Epoch")
_ = axes[1].set_xlabel("Epoch")
_ = axes[0].set_title("Training loss")
_ = axes[1].set_title("Validation RMSE")

fig.savefig(os.path.join(deepsensor_folder, "patchwise_training_loss.png"))


# prediction with patches ON-GRID, select one data from the validation tasks
# generate patchwise tasks for a specific date
# pick a random date as datetime64[ns]

dates = [np.datetime64("2019-06-25")]
eval_task = gen_validation_tasks(dates, progress=False)
# test_task = task_loader(date, [100, "all", "all"], seed_override=42)
pred = model.predict_patch(eval_task, data_processor=data_processor, stride_size=(1, 1), patch_size=(0.5, 0.5), X_t=era5_raw_ds, resolution_factor=2)

import pdb
pdb.set_trace()

fig = deepsensor.plot.prediction(pred, dates[0], data_processor, task_loader, eval_task[0], crs=ccrs.PlateCarree())
fig.savefig(os.path.join(deepsensor_folder, "patchwise_prediction.png"))

print(0)


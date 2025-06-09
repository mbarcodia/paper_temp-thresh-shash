"""Trainer modules for pytorch models.

Classes
---------
Trainer(base.base_trainer.BaseTrainer)

"""

import numpy as np
import torch
from base.base_trainer import BaseTrainer
from base.base_transfer_trainer import (
    BaseTransferTrainer,
)  # Import TransferTrainer class
from utils import MetricTracker
import model.metric as module_metric


class TransferTrainer(BaseTransferTrainer):
    """
    Trainer class that uses TransferTrainer but incorporates data loaders and scheduling.
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        data_loader,
        validation_data_loader,
        device,
        config,
        scheduler=None,
    ):
        super().__init__(model, criterion, metric_funcs, optimizer, max_epochs, config)
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        # self.do_validation = True
        # Set do_validation flag based on whether validation_data_loader is provided
        self.do_validation = validation_data_loader is not None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.batch_log.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            input, input_unit, input_co2, target = (
                data[0].to(self.device),
                data[1].to(self.device),
                data[2].to(self.device),
                target.to(self.device),
            )

            self.optimizer.zero_grad()
            output = self.model(input, input_unit, input_co2)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Log results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            self.batch_log.update(
                "train_custom_mae", module_metric.custom_mae(output, target)
            )  # Custom MAE

            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

        # # Check early stopping based on training MAE
        # if self.early_stopper.check_early_stop(
        #     epoch, self.batch_log.history["train_custom_mae"][epoch], self.model
        # ):
        #     print(
        #         f"Restoring model weights from the best epoch {self.early_stopper.best_epoch}: "
        #         f"train_custom_mae = {self.early_stopper.min_validation_loss:.5f}"
        #     )
        #     self.model.load_state_dict(self.early_stopper.best_model_state)
        #     self.model.eval()
        #     return  # Stop training

    def _validation_epoch(self, epoch):
        """No validation phase since we only use training loss for early stopping."""
        pass

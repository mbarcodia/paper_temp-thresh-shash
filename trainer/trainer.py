"""Trainer modules for pytorch models.

Classes
---------
Trainer(base.base_trainer.BaseTrainer)

"""

import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
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
        scheduler=None,  # Add scheduler as an optional parameter
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs,
            config,
        )
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

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(input, input_unit, input_co2)
            # output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

        # Run validation
        if self.do_validation:
            self._validation_epoch(epoch)

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        if not self.validation_data_loader:  # Ensure it's not None before iterating
            return

        self.model.eval()
        with torch.inference_mode():

            for batch_idx, (data, target) in enumerate(self.validation_data_loader):
                input, input_unit, input_co2, target = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    target.to(self.device),
                )

                output = self.model(input, input_unit, input_co2)
                loss = self.criterion(output, target)

                # Log the results
                self.batch_log.update("val_loss", loss.item())
                for met in self.metric_funcs:
                    self.batch_log.update("val_" + met.__name__, met(output, target))

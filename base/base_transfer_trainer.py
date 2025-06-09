import torch
import time
import copy
from utils import MetricTracker
from abc import abstractmethod

import model.metric as module_metric


class BaseTransferTrainer:
    """
    Trainer for transfer learning with early stopping based only on training custom MAE.
    """

    def __init__(self, model, criterion, metric_funcs, optimizer, max_epochs, config):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.max_epochs = max_epochs

        # Use EarlyStoppingTransfer which monitors train_custom_mae
        self.early_stopper = EarlyStoppingTransfer(
            **config["trainer"]["early_stopping"]["args"]
        )

        self.metric_funcs = metric_funcs

        # Modify logs: remove validation metrics
        self.batch_log = MetricTracker(
            "batch",
            "loss",
            "train_custom_mae",
            *[m.__name__ for m in self.metric_funcs],
        )
        self.log = MetricTracker(
            "epoch",
            "loss",
            "train_custom_mae",
            *[m.__name__ for m in self.metric_funcs],
        )

    def fit(self):
        """
        Full training logic with early stopping based on training custom MAE.
        """
        for epoch in range(self.max_epochs + 1):

            start_time = time.time()

            self._train_epoch(epoch)

            # Log results of the epoch
            self.batch_log.result()
            self.log.update("epoch", epoch)
            for key in self.batch_log.history:
                self.log.update(key, self.batch_log.history[key])

            # Update scheduler using training custom MAE
            # if self.scheduler is not None:
            #     self.scheduler.step(self.log.history["train_custom_mae"][epoch])
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if (
                        "train_loss" in self.log.history
                        and len(self.log.history["train_loss"]) > 0
                    ):
                        self.scheduler.step(
                            self.log.history["train_loss"][-1]
                        )  # For ReduceLROnPlateau
                else:
                    self.scheduler.step()  # For StepLR and other schedulers

            for param_group in self.optimizer.param_groups:
                print(f"Epoch {epoch}: Learning rate = {param_group['lr']}")

            # Early stopping based on training custom MAE
            if self.early_stopper.check_early_stop(
                epoch, self.log.history["train_custom_mae"][epoch], self.model
            ):
                print(
                    f"Restoring model weights from the best epoch {self.early_stopper.best_epoch}: "
                    f"train_custom_mae = {self.early_stopper.min_training_mae:.5f}"
                )
                self.log.print(idx=self.early_stopper.best_epoch)

                self.model.load_state_dict(self.early_stopper.best_model_state)
                self.model.eval()
                break

            # Print out progress during training
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Epoch {epoch:3d}/{self.max_epochs:2d}\n"
                f"  {elapsed_time:.1f}s"
                f" - train_loss: {self.log.history['loss'][epoch]:.5f}"
                f" - train_custom_mae: {self.log.history['train_custom_mae'][epoch]:.5f}"
            )

        # Reset the batch_log
        self.batch_log.reset()
        self.model.eval()

    @abstractmethod
    def _train_epoch(self):
        """
        Train an epoch.
        """
        raise NotImplementedError


class EarlyStoppingTransfer:
    """
    Early stopping based only on training custom MAE.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_mae = float("inf")
        self.best_model_state = None
        self.best_epoch = None

    def check_early_stop(self, epoch, training_mae, model):
        """
        Stop training if the training custom MAE does not improve by at least min_delta for 'patience' epochs.
        """
        if training_mae < (self.min_training_mae - self.min_delta):
            self.min_training_mae = training_mae
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
import pathlib
import time

import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import json

from .base import Experiment
from .. import datasets
from .. import models
from ..metrics import correct
from ..models.head import mark_classifier
from ..util import printc, OnlineStats


class TrainingExperiment(Experiment):

    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'optim_kwargs': {'lr': 1e-3}
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 finetune=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        # Default children kwargs
        super(TrainingExperiment, self).__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs
        self.add_params(**params)
        # Save params

        self.build_dataloader(dataset, **dl_kwargs)

        self.build_model(model, pretrained, resume)

        self.build_train(resume_optim=resume_optim, **train_kwargs)

        self.path = path
        self.save_freq = save_freq

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)
        self.run_epochs()

    def build_dataloader(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        if 'cifar_shape' in dl_kwargs:
            cifar_shape = dl_kwargs.pop('cifar_shape')
            self.train_dataset = constructor(train=True, cifar_shape=cifar_shape)
            self.val_dataset = constructor(train=False, cifar_shape=cifar_shape)
        else:
            self.train_dataset = constructor(train=True)
            self.val_dataset = constructor(train=False)

        # verification dataset used for per layer fraction selection
        verif_ind = [] if self.params['finetune'] else \
            list(range(0, len(self.train_dataset), int(len(self.train_dataset)/len(self.val_dataset))))
        train_ind = list(set(range(0, len(self.train_dataset))) - set(verif_ind))
        self.verif_dataset = torch.utils.data.Subset(self.train_dataset, verif_ind)
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_ind)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        self.verif_dl = DataLoader(self.verif_dataset, shuffle=False, **dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)

    def build_model(self, model, pretrained=True, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

    def build_train(self, optim, epochs, resume_optim=False, scheduler=None, **train_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3},
            'Adam': {'betas': (.9, .99), 'lr': 1e-4}  # JG removed 'momentum': 0.9,
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **train_kwargs['optim_kwargs']}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        # scheduler
        if isinstance(scheduler, str):
            constructor = getattr(torch.optim.lr_scheduler, scheduler)
            scheduler = constructor(self.optim, **train_kwargs['scheduler_kwargs'])
        self.scheduler = scheduler

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{epoch}.pt')

    def run_epochs(self):

        since = time.perf_counter()
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                total_loss_mean, acc1_mean, acc5_mean = self.eval(epoch)
                if self.scheduler is not None:
                    self.scheduler.step()
                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                # TODO Early stopping based on desired error input
                if acc1_mean > 0.99:
                    self.checkpoint()
                    break
                # TODO ReduceLR on plateau?
                self.log(timestamp=time.perf_counter() - since)
                self.log_epoch(epoch)


        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def run_epoch(self, train, opt=True, epoch=0, verif=False):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            prefix = 'val'
            dl = self.verif_dl if verif else self.val_dl
            self.model.eval()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train and opt:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        })

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch=0):
        return self.run_epoch(True, epoch=epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch=epoch)

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5',
                ]

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__
        
        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)

import argparse
import model as M
import nnue_dataset
import nnue_bin_dataset
import nnue_data_module
import lightning.pytorch as pl
import features
import os
import torch
from torch import set_num_threads as t_set_num_threads
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from torch.utils.data import DataLoader, Dataset


def cli_main():
  torch.set_float32_matmul_precision('high')
  cli = LightningCLI(M.NNUE, nnue_data_module.NNUEDataModule)

if __name__ == '__main__':
  cli_main()

import model as M
import nnue_data_module
import torch
from lightning.pytorch.cli import LightningCLI


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(M.NNUE, nnue_data_module.NNUEDataModule)


if __name__ == "__main__":
    cli_main()

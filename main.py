from utils.logger import make_logger
import argparse
import torch
import os
import numpy as np

from tqdm import tqdm
from data.load_data import DataFactory

from utils.tools import SEED_everything
SEED_everything(42)

parser = argparse.ArgumentParser(description='THOC-Pytorch')
# data
parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--eval_batch_size", type=int, required=False, default=64 * 3, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required=False, default=100, help=f"window size")
parser.add_argument("--stride", type=int, required=False, default=1, help=f"stride")
parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
parser.add_argument("--load_pretrained", action="store_true", help=f"whether to load pretrained version")
parser.add_argument("--exp_id", type=str, default="test")
parser.add_argument("--scaler", type=str, default="std")
parser.add_argument("--window_anomaly", action="store_true", help=f"window-base anomaly")
parser.add_argument("--eval_every_epoch", action="store_true", help=f"evaluate every epoch")
parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

# save
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--logs", type=str, default="./logs")
parser.add_argument("--outputs", type=str, default="./outputs")

args = parser.parse_args()
args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
args.output_path = os.path.join(args.outputs, f"{args.exp_id}")

os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.logging_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)

args.home_dir = "."
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
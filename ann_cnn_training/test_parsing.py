import argparse
import numpy as np
import logging
import sys

sys.path.append("../utils/")
from dataloader_definition import Dataset_ANN_CNN

from test_parsing2 import test_fn

logger = logging.getLogger(__name__)

logging.basicConfig(filename="log.txt", level=logging.INFO)
logger.info("Started")
test_fn()
logger.info("Finished")


parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--input", type=int, choices=[0, 1, 2], default=1,
#                    help="increase output verbosity")
parser.add_argument(
    "-d",
    "--horizontal",
    choices=["global"],
    default="global",
    help="Horizontal domain for training",
)
parser.add_argument(
    "-v",
    "--vertical",
    choices=["global", "stratosphere_update"],
    default="global",
    help="Vertical domain for training",
)
parser.add_argument(
    "-f",
    "--features",
    choices=["uvtheta", "uvw", "uvthetaw"],
    default="uvtheta",
    help="Feature set for training",
)
parser.add_argument(
    "-s", "--stencil", type=int, choices=[1, 3, 5], default=1, help="Horizontal stencil for the NN"
)
parser.add_argument("-i", "--input_dir", default=".", help="Input directory with training data")
parser.add_argument("-o", "--output_dir", default=".", help="Output directory to store checkpoints")
args = parser.parse_args()


print(f"horizontal={args.horizontal}")
print(f"vertical={args.vertical}")
print(f"features={args.features}")
print(f"stencil={args.stencil}")
print(f"input_dir={args.input_dir}")
print(f"output_dir={args.output_dir}")

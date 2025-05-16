import torch
import sys # Import sys to potentially access command line args, though parse_args() does this by default
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def get_main_args():
    """
    Parses command-line arguments for the main script.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    # Data and Directory arguments
    arg("--base_dir", type=str, default="LiveEO_ML_intern_challenge", help="Base directory containing the 'data' folder.")
    arg("--save_path", type=str, default='./', help='Path to save checkpoints and predictions.')

    # Model arguments
    arg ("--in_channels", type=int, default=3, help="#Input Channels (e.g., RGB)")
    arg ("--out_channels", type=int, default=1, help="#Output Channels (e.g., building mask)")
    arg ("--kernels",  default=[[3, 3]] * 5, help="Convolution Kernels for DynUNet") # Note: This default might need adjustment based on actual DynUNet usage
    arg ("--strides",  default=[[1, 1]] +  [[2, 2]] * 4, help="Convolution Strides for DynUNet") # Note: This default might need adjustment

    # Training arguments
    arg ("--seed", type=int, default=26012022, help="Random Seed for reproducibility")
    # The generator argument default might be problematic if not handled carefully by PyTorch Lightning
    # It's often better to just set the seed and let PL handle the generator
    # arg("--generator", default=torch.Generator().manual_seed(26012022), help='Train Validate Predict Seed')
    arg("--num_epochs", type=int, default=20, help="Number of training epochs")
    arg("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    arg ("--weigh_decay", type=float, default=1e-5, help="Optimizer weight decay")
    arg("--samples_per_epoch", type=int, default=1000, help="Number of random samples to use per training epoch")
    arg("--crop_size", type=int, default=480, help="Size of random crops for training images")
    arg("--batch_size", type=int, default=12, help="Batch size for DataLoaders")
    arg ("--num_workers", type=int, default=2, help="Number of worker processes for DataLoaders")

    # Execution mode arguments
    arg("--exec_mode", type=str, default='train', choices=['train', 'predict'], help='Execution Mode (train or predict)')
    arg("--ckpt_path", type=str, default=None, help='Path to a checkpoint file for resuming training or prediction')

    # Parse the actual command-line arguments
    # Remove the args=[] to parse sys.argv
    return parser.parse_args()


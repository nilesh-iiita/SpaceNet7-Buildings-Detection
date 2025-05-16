from args import get_main_args # Assuming get_main_args is in args.py
from UNet_monai import Unet # Assuming Unet is in UNet_monai.py
from pytorch_lightning import Trainer, seed_everything # Corrected import for seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet7DataModule # Assuming SpaceNet7DataModule is in dataset.py
import torch # Imported torch for potential use, though not strictly needed in this snippet


if __name__ == "__main__":
    args = get_main_args()

    # Set random seed for reproducibility
    seed_everything(args.seed)

    callbacks = []
    model = Unet(args)

    # Define ModelCheckpoint callback
    # Ensure the directory exists if not using the default './'
    model_ckpt = ModelCheckpoint(
        dirpath=args.save_path, # Use args.save_path for checkpoint directory
        filename="best_model",
        monitor="dice_mean", # Make sure 'dice_mean' is logged by your validation_epoch_end hook
        mode="max",
        save_last=True
    )
    callbacks.append(model_ckpt)

    # Initialize DataModule
    dm = SpaceNet7DataModule(args)

    # Initialize Trainer
    # Added precision=16 for Automatic Mixed Precision (AMP)
    # Removed profiler='simple' as it might be deprecated or require a callback object
    trainer = Trainer(
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=args.num_epochs,
        enable_progress_bar=True,
        accelerator="gpu", # Use 'gpu' accelerator
        devices=1, # Use 1 GPU
        precision=16, # Enable 16-bit mixed precision training to reduce memory usage
        # profiler='simple' # Removed this argument
    )

    # train the model
    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    else:
        # For prediction, ensure ckpt_path is provided in args
        if args.ckpt_path is None:
            print("Error: Checkpoint path (--ckpt_path) must be provided for prediction mode.")
            exit() # Exit if checkpoint path is missing for prediction

        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)





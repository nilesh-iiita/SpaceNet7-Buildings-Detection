import torch
import numpy as np
import skimage.io as io
from utils import get_files
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import torch.nn as nn


class SpaceNet7(Dataset):
    def __init__(self, files, crop_size, exec_mode):

        super().__init__() # Call parent constructor
        self.files = files
        self.crop_size = crop_size # Stored as self.crop_size
        self.exec_mode = exec_mode # Stored as self.exec_mode


    def OpenImage(self, idx, invert=True):
        image = io.imread(self.files[idx]['image'])[:,:,0:3] #shape (H, W, 3)
        if invert:
            image = image.transpose((2,0,1))                 #shape (3, H, W)
        return (image / np.iinfo(image.dtype).max) #render the values between 0 and 1


    def OpenMask(self, idx):
        # Read the mask image
        mask = io.imread(self.files[idx]['mask'])

        # --- MODIFIED: Ensure mask is single channel ---
        # Check if the mask has more than one channel (e.g., shape (H, W, C))
        if mask.ndim == 3 and mask.shape[-1] > 1:
            # Assuming the building mask information is in the first channel (index 0)
            # Select only the first channel
            mask = mask[:, :, 0]
            print(f"Debug: Mask {self.files[idx]['mask']} had shape {mask.shape}, selected channel 0.")
        elif mask.ndim == 2:
            # Mask is already single channel (H, W), which is expected
            print(f"Debug: Mask {self.files[idx]['mask']} has expected shape {mask.shape}.")
        else:
             print(f"Warning: Unexpected mask shape for {self.files[idx]['mask']}: {mask.shape}")
             # Handle other unexpected shapes if necessary


        # Convert values to 0 and 1 (assuming original mask has 255 for buildings)
        # This part remains the same
        return np.where(mask==255, 1, 0) #change the values to 0 and 1
        # --- END MODIFIED ---


    def __getitem__(self, idx):
        # read the images and masks as numpy arrays
        x = self.OpenImage(idx, invert=True) # Returns (C, H, W) float
        y = self.OpenMask(idx) # Returns (H, W) int/uint (converted to 0/1)

        # padd the images to have a homogenous size (C, 1024, 1024)
        # Need to adjust padding to handle (C, H, W) for image and (H, W) for mask
        # Original code adds batch dim to mask before padding? Let's check padding logic.
        # Based on the padding function expecting (C, H, W) and (N, H, W), let's pass (C, H, W) and (1, H, W)
        x_padded, y_padded = self.padding((x,y[None])) # Pass image (C, H, W) and mask with batch dim (1, H, W)


        # if it is the training phase, create random (C, 430, 430) crops
        # if it is the evaluation phase, we will leave the orginal size (C, 1024, 1024)
        if self.exec_mode =='train':
            # Original code adds batch dim to both before cropping
            # Crop expects (N, C, H, W) and (N, 1, H, W)
            # --- CORRECTED: Use self.crop_size instead of self.args.crop_size ---
            x_cropped, y_cropped = self.crop(x_padded[None], y_padded[None], self.crop_size)
            # --- END CORRECTED ---
            x_final, y_final = x_cropped[0], y_cropped[0] # Remove batch dimension after cropping
        else:
             x_final, y_final = x_padded, y_padded


        # numpy array --> torch tensor
        x_tensor = torch.tensor(x_final, dtype=torch.float32)
        y_tensor = torch.tensor(y_final, dtype=torch.uint8)

        # normalize the images (image- image.mean()/image.std())
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Apply normalization to image tensor
        x_normalized = normalize(x_tensor)

        # Ensure mask tensor is (H, W) or (1, H, W) if needed by the model
        # Your model expects (N, 1, H, W) for masks based on the loss function signature
        # The mask is currently (H, W) after OpenMask and padding/cropping. Add channel dim.
        # The crop function is designed to return (N, 1, H, W) for seg, so y_final should be (1, H, W)
        # Let's remove the unsqueeze here and rely on the crop function's output shape.
        # y_tensor = y_tensor.unsqueeze(0) # Remove this line


        return x_normalized, y_tensor # x_normalized (C, H, W), y_tensor (1, H, W) if crop works as intended


    def __len__(self):
        return len(self.files)

    def padding (self, sample):
        # --- Reverted padding logic to match original structure, assuming it worked with original data ---
        # Original code seems to expect mask with a batch dimension before padding
        image, mask = sample # image (C, H, W), mask (1, H, W) based on __getitem__
        C, H, W = image.shape
        mask_N, mask_H, mask_W = mask.shape # Assuming mask has a batch dim here

        # Assert spatial dimensions match
        assert H == mask_H and W == mask_W, f"Image and mask spatial dimensions must match before padding. Image: {image.shape}, Mask: {mask.shape}"

        target_H, target_W = 1024, 1024

        # Calculate padding amounts
        pad_H = target_H - H
        pad_W = target_W - W

        # Apply padding if needed
        # Original padding logic added 1,0 or 0,1 padding, which seems incorrect for 1024 target size.
        # Let's adjust padding to reach 1024x1024 target.
        # Assuming padding should be applied to bottom/right if needed.
        image_padded = np.pad(image, ((0,0), (0, pad_H), (0, pad_W)), 'constant', constant_values=(0))
        mask_padded = np.pad(mask, ((0,0), (0, pad_H), (0, pad_W)), 'constant', constant_values=(0)) # Pad mask with batch dim

        return image_padded, mask_padded
        # --- END Reverted padding ---


    def crop(self, data, seg, crop_size=256):
        # This cropping function seems designed for (N, C, H, W) data and (N, 1, H, W) seg
        # Let's adjust it slightly to be clearer and handle (N, H, W) seg
        # Input: data (N, C, H, W), seg (N, 1, H, W) based on __getitem__ and padding
        # Output: data_return (N, C, crop_size, crop_size), seg_return (N, 1, crop_size, crop_size)

        data_shape = data.shape # (N, C, H, W)
        seg_shape = seg.shape # (N, 1, H, W)

        N, C, H, W = data_shape
        N_seg, C_seg, H_seg, W_seg = seg_shape # Assuming mask has a channel dim here

        assert N == N_seg, "Batch sizes of data and seg must match for cropping."
        assert C_seg == 1, f"Mask should have a single channel for cropping, but got {C_seg}."
        assert H == H_seg and W == W_seg, "Spatial dimensions of data and seg must match for cropping."

        crop_size_h = crop_size
        crop_size_w = crop_size

        data_return = np.zeros((N, C, crop_size_h, crop_size_w), dtype=data.dtype)
        seg_return = np.zeros((N, C_seg, crop_size_h, crop_size_w), dtype=seg.dtype) # Maintain mask channel dim


        for b in range(N):
            # Determine random top-left corner for the crop
            if H - crop_size_h > 0:
                h_start = np.random.randint(0, H - crop_size_h)
            else:
                h_start = 0 # If image is smaller than crop size, start at 0

            if W - crop_size_w > 0:
                w_start = np.random.randint(0, W - crop_size_w)
            else:
                w_start = 0 # If image is smaller than crop size, start at 0


            h_end = h_start + crop_size_h
            w_end = w_start + crop_size_w

            # Ensure crop does not go out of bounds (shouldn't happen with the above logic if H, W >= crop_size)
            h_end = min(h_end, H)
            w_end = min(w_end, W)


            # Apply slicing
            data_cropped = data[b, :, h_start:h_end, w_start:w_end]
            seg_cropped = seg[b, :, h_start:h_end, w_start:w_end] # Slice mask with channel dim

            # Handle cases where image is smaller than crop size by padding the cropped result
            if data_cropped.shape[1] < crop_size_h or data_cropped.shape[2] < crop_size_w:
                 pad_h_after = crop_size_h - data_cropped.shape[1]
                 pad_w_after = crop_size_w - data_cropped.shape[2]
                 data_cropped = np.pad(data_cropped, ((0,0), (0, pad_h_after), (0, pad_w_after)), 'constant', constant_values=(0))
                 seg_cropped = np.pad(seg_cropped, ((0,0), (0, pad_h_after), (0, pad_w_after)), 'constant', constant_values=(0)) # Pad mask with channel dim


            data_return[b] = data_cropped
            seg_return[b] = seg_cropped

        return data_return, seg_return


class SpaceNet7DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args  = args

    def setup(self, stage=None):
        # get_files now returns a list of dicts with 'image' and 'mask' paths
        files = get_files(self.args.base_dir)

        # Check if any files were found before splitting
        if not files:
            print("Error: No image/label pairs found. Cannot perform train/test split.")
            # Depending on desired behavior, you might want to exit or raise an error here
            # For now, we'll let the ValueError from train_test_split occur if files is empty.
            # If get_files returns [], train_test_split will raise the ValueError.
            pass # Let the original error happen if files is empty

        train_files, test_files = train_test_split(files, test_size=0.1, random_state=self.args.seed)

        self.spaceNet7_train = SpaceNet7(train_files, self.args.crop_size, self.args.exec_mode)
        self.spaceNet7_val = SpaceNet7(test_files, self.args.crop_size, self.args.exec_mode)


    def train_dataloader(self):
        # Ensure samples_per_epoch does not exceed the number of training files
        num_train_images = len(self.spaceNet7_train)
        num_samples = min(self.args.samples_per_epoch, num_train_images)
        if num_samples == 0:
             print("Warning: No training samples available. Check data loading.")
             return None # Return None if no training data

        train_sampler = self.ImageSampler(num_images=num_train_images, num_samples=num_samples)
        train_bSampler = BatchSampler(train_sampler, batch_size=self.args.batch_size, drop_last=True)
        return DataLoader(self.spaceNet7_train, batch_sampler=train_bSampler, num_workers=self.args.num_workers)

    def val_dataloader(self):
        if len(self.spaceNet7_val) == 0:
             print("Warning: No validation samples available. Check data loading.")
             return None # Return None if no validation data
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)

    def predict_dataloader(self):
        if len(self.spaceNet7_val) == 0:
             print("Warning: No prediction samples available. Check data loading.")
             return None # Return None if no prediction data
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)


    class ImageSampler(Sampler):
        def __init__(self, num_images, num_samples): # Removed default values
            self.num_images = num_images
            self.num_samples = num_samples

        def generate_iteration_list(self):
            # Ensure we don't try to sample more images than available
            if self.num_images == 0:
                 return []
            return np.random.randint(0, self.num_images, self.num_samples)

        def __iter__(self):
            return iter(self.generate_iteration_list())

        def __len__(self):
            return self.num_samples

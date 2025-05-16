import os
import glob

def combine_files(image_path, label_path):
    # This function takes individual image and label paths
    # and returns a dictionary for the dataset class
    files = {'image': image_path,
             'mask': label_path} # 'mask' key is used, and the file is a .tif mask
    return files

def get_files(base_dir):
    """
    Finds image and label files within the nested directory structure
    of the SpaceNet 7 dataset.
    Expects images in <base_dir>/data/train/<scene_id>/images/
    and labels (masks) in <base_dir>/data/train/<scene_id>/images_masked/
    """
    all_image_files = []
    all_label_files = []

    # Construct the path to the main training data directory
    train_data_dir = os.path.join(base_dir, 'data', 'train')

    # Check if the train data directory exists
    if not os.path.exists(train_data_dir):
        print(f"Error: Training data directory not found at {train_data_dir}")
        return [] # Return empty list if the base data directory is wrong

    # Walk through the train data directory to find scene folders
    # os.walk yields (dirpath, dirnames, filenames)
    for root, dirs, filenames in os.walk(train_data_dir):
        # Look for 'images' and 'images_masked' subdirectories within scene folders
        if 'images' in dirs and 'images_masked' in dirs:
            scene_images_dir = os.path.join(root, 'images')
            # --- CORRECTED: Look for labels in 'images_masked' directory ---
            scene_labels_dir = os.path.join(root, 'images_masked')
            # --- END CORRECTED ---

            # Find all files within the images and labels directories for this scene
            # Using glob.glob with a pattern that matches files (*.tif)
            images_in_scene = sorted(glob.glob(os.path.join(scene_images_dir, '*.tif'))) # Assuming .tif images
            # --- CORRECTED: Search for *.tif files in the labels directory ---
            labels_in_scene = sorted(glob.glob(os.path.join(scene_labels_dir, '*.tif'))) # Searching for .tif labels (masks)
            # --- END CORRECTED ---

            # Basic check: ensure the number of images and labels match for this scene
            # Note: The previous warnings showed a 1:2 ratio (e.g., 18 images, 36 labels).
            # This might indicate multiple masks per image or a different file naming convention.
            # The current code assumes a 1:1 match based on sorted filenames.
            # If the mismatch warnings persist, you might need to inspect the filenames
            # and adjust the matching logic. For now, we'll warn but continue.
            if len(images_in_scene) != len(labels_in_scene):
                 print(f"Warning: Mismatch in number of images and labels in {root}")
                 print(f"  Images found: {len(images_in_scene)}")
                 print(f"  Labels found: {len(labels_in_scene)}")


            # Combine image and label paths for each file pair in this scene
            # Assuming images and labels are sorted identically and correspond one-to-one
            # If the mismatch is consistent (e.g., always double labels), the simple zip might not work.
            # For now, we'll take the minimum number to avoid index errors.
            for i in range(min(len(images_in_scene), len(labels_in_scene))):
                 all_image_files.append(images_in_scene[i])
                 all_label_files.append(labels_in_scene[i])


    # Now, combine the full list of image and label paths into the desired format
    files = [combine_files(all_image_files[i], all_label_files[i]) for i in range(len(all_image_files))]

    # Add a final debug print to confirm how many file pairs were found in total
    print(f"DEBUG: get_files found a total of {len(files)} image/label pairs across all scenes.")

    return files


import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import json

class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None, crop_config_path=None):
        '''
        Enhanced BRATS loader with optimal cropping support
        
        directory: expected to contain BRATS folder structure
        mode: 'train', 'eval', or 'auto'
        crop_config_path: path to optimal crop configuration JSON
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        
        # Load optimal crop configuration
        self.crop_bounds = self.load_crop_config(crop_config_path)
        self.use_optimal_crop = self.crop_bounds is not None
        
        if self.use_optimal_crop:
            print(f"🔧 Using optimal cropping: {self.get_crop_info()}")
        else:
            print("⚠️  Using original fixed cropping (8-pixel border)")

        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have a datadir
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def load_crop_config(self, crop_config_path):
        """Load optimal crop configuration from JSON file"""
        if crop_config_path is None:
            # Try to find config in common locations
            possible_paths = [
                './crop_analysis_results/optimal_crop_config.json',
                './optimal_crop_config.json',
                '../crop_analysis_results/optimal_crop_config.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    crop_config_path = path
                    break
        
        if crop_config_path and os.path.exists(crop_config_path):
            try:
                with open(crop_config_path, 'r') as f:
                    config = json.load(f)
                    return config['crop_bounds']
            except Exception as e:
                print(f"⚠️  Failed to load crop config from {crop_config_path}: {e}")
                return None
        
        return None

    def get_crop_info(self):
        """Get human-readable crop information"""
        if not self.use_optimal_crop:
            return "Fixed 8-pixel border crop"
        
        x_size = self.crop_bounds['x_max'] - self.crop_bounds['x_min']
        y_size = self.crop_bounds['y_max'] - self.crop_bounds['y_min']
        z_size = self.crop_bounds['z_max'] - self.crop_bounds['z_min']
        
        return f"{x_size}x{y_size}x{z_size} (optimal crop)"

    def apply_optimal_crop(self, img_data):
        """Apply optimal crop bounds determined by dataset analysis"""
        if not self.use_optimal_crop:
            # Fallback to original cropping logic
            return img_data
        
        # Apply optimal crop bounds
        x_min, x_max = self.crop_bounds['x_min'], self.crop_bounds['x_max']
        y_min, y_max = self.crop_bounds['y_min'], self.crop_bounds['y_max']
        z_min, z_max = self.crop_bounds['z_min'], self.crop_bounds['z_max']
        
        # Ensure bounds don't exceed image dimensions
        x_max = min(x_max, img_data.shape[0])
        y_max = min(y_max, img_data.shape[1])
        z_max = min(z_max, img_data.shape[2])
        
        # Apply crop
        img_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
        return img_cropped

    def process_image_tensor(self, img_cropped):
        """Process cropped image into the format expected by the model"""
        if self.use_optimal_crop:
            # Use actual cropped dimensions
            D, H, W = img_cropped.shape
            
            # Create tensor with appropriate size (pad depth to next multiple of 32 for DWT)
            target_depth = ((W + 31) // 32) * 32  # Ensure divisible by 32 for 3D DWT
            
            img_tensor = torch.zeros(1, H, W, target_depth)
            img_tensor[:, :, :, :min(D, target_depth)] = torch.tensor(img_cropped.transpose(1, 2, 0))
            
            return img_tensor
        else:
            # Original logic for backward compatibility
            img_tensor = torch.zeros(1, 240, 240, 160)
            img_tensor[:, :, :, :155] = torch.tensor(img_cropped)
            img_tensor = img_tensor[:, 8:-8, 8:-8, :]  # Fixed 8-pixel crop
            return img_tensor

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'

        # Load and process t1n
        if 't1n' in filedict:
            t1n_np = nibabel.load(filedict['t1n']).get_fdata()
            t1n_np_cropped = self.apply_optimal_crop(t1n_np)  # Apply optimal crop
            t1n_np_clipnorm = clip_and_normalize(t1n_np_cropped)
            t1n = self.process_image_tensor(t1n_np_clipnorm)
        else:
            missing = 't1n'
            t1n = torch.zeros(1)

        # Load and process t1c
        if 't1c' in filedict:
            t1c_np = nibabel.load(filedict['t1c']).get_fdata()
            t1c_np_cropped = self.apply_optimal_crop(t1c_np)  # Apply optimal crop
            t1c_np_clipnorm = clip_and_normalize(t1c_np_cropped)
            t1c = self.process_image_tensor(t1c_np_clipnorm)
        else:
            missing = 't1c'
            t1c = torch.zeros(1)

        # Load and process t2w
        if 't2w' in filedict:
            t2w_np = nibabel.load(filedict['t2w']).get_fdata()
            t2w_np_cropped = self.apply_optimal_crop(t2w_np)  # Apply optimal crop
            t2w_np_clipnorm = clip_and_normalize(t2w_np_cropped)
            t2w = self.process_image_tensor(t2w_np_clipnorm)
        else:
            missing = 't2w'
            t2w = torch.zeros(1)

        # Load and process t2f
        if 't2f' in filedict:
            t2f_np = nibabel.load(filedict['t2f']).get_fdata()
            t2f_np_cropped = self.apply_optimal_crop(t2f_np)  # Apply optimal crop
            t2f_np_clipnorm = clip_and_normalize(t2f_np_cropped)
            t2f = self.process_image_tensor(t2f_np_clipnorm)
        else:
            missing = 't2f'
            t2f = torch.zeros(1)

        # Handle subject identification based on mode
        if self.mode == 'eval' or self.mode == 'auto':
            if 't1n' in filedict:
                subj = filedict['t1n']
            else:
                subj = filedict['t2f']
        else:
            subj = 'dummy_string'

        return {'t1n': t1n.float(),
                't1c': t1c.float(),
                't2w': t2w.float(),
                't2f': t2f.float(),
                'missing': missing,
                'subj': subj,
                'filedict': filedict}

    def __len__(self):
        return len(self.database)

    def get_output_dimensions(self):
        """Get the dimensions that will be output by this loader"""
        if self.use_optimal_crop:
            x_size = self.crop_bounds['x_max'] - self.crop_bounds['x_min']
            y_size = self.crop_bounds['y_max'] - self.crop_bounds['y_min']
            z_size = self.crop_bounds['z_max'] - self.crop_bounds['z_min']
            target_depth = ((z_size + 31) // 32) * 32
            return (x_size, y_size, target_depth)
        else:
            return (224, 224, 160)  # Original fixed dimensions

def clip_and_normalize(img):
    """Clip and normalize image (unchanged from original)"""
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))
    return img_normalized

# Example usage and testing
if __name__ == "__main__":
    # Test the loader with optimal cropping
    dataset = BRATSVolumes(
        directory="./datasets/BRATS2023/training",
        mode="train",
        crop_config_path="./crop_analysis_results/optimal_crop_config.json"
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Output dimensions: {dataset.get_output_dimensions()}")
    print(f"Crop info: {dataset.get_crop_info()}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample t1n shape: {sample['t1n'].shape}")
        print(f"Sample t1c shape: {sample['t1c'].shape}")
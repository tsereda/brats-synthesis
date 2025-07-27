import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel

class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        
        # DWT-compatible crop bounds: (160, 208, 152) - all even dimensions
        self.crop_bounds = {
            'x_min': 39, 'x_max': 199,  # width: 160
            'y_min': 17, 'y_max': 225,  # height: 208
            'z_min': 0,  'z_max': 152   # depth: 152
        }
        
        print(f"🔧 Using crop: 160x208x152 (DWT-compatible)")

        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:  # leaf directory with actual files
                files.sort()
                datapoint = dict()
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def apply_crop(self, img_data):
        """Apply fixed crop bounds"""
        x_min, x_max = self.crop_bounds['x_min'], self.crop_bounds['x_max']
        y_min, y_max = self.crop_bounds['y_min'], self.crop_bounds['y_max']
        z_min, z_max = self.crop_bounds['z_min'], self.crop_bounds['z_max']
        
        return img_data[x_min:x_max, y_min:y_max, z_min:z_max]

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'

        # Load and process t1n
        if 't1n' in filedict:
            t1n_np = nibabel.load(filedict['t1n']).get_fdata()
            t1n_cropped = self.apply_crop(t1n_np)
            t1n_norm = clip_and_normalize(t1n_cropped)
            t1n = torch.tensor(t1n_norm).float().unsqueeze(0)
        else:
            missing = 't1n'
            t1n = torch.zeros(1)

        # Load and process t1c
        if 't1c' in filedict:
            t1c_np = nibabel.load(filedict['t1c']).get_fdata()
            t1c_cropped = self.apply_crop(t1c_np)
            t1c_norm = clip_and_normalize(t1c_cropped)
            t1c = torch.tensor(t1c_norm).float().unsqueeze(0)
        else:
            missing = 't1c'
            t1c = torch.zeros(1)

        # Load and process t2w
        if 't2w' in filedict:
            t2w_np = nibabel.load(filedict['t2w']).get_fdata()
            t2w_cropped = self.apply_crop(t2w_np)
            t2w_norm = clip_and_normalize(t2w_cropped)
            t2w = torch.tensor(t2w_norm).float().unsqueeze(0)
        else:
            missing = 't2w'
            t2w = torch.zeros(1)

        # Load and process t2f
        if 't2f' in filedict:
            t2f_np = nibabel.load(filedict['t2f']).get_fdata()
            t2f_cropped = self.apply_crop(t2f_np)
            t2f_norm = clip_and_normalize(t2f_cropped)
            t2f = torch.tensor(t2f_norm).float().unsqueeze(0)
        else:
            missing = 't2f'
            t2f = torch.zeros(1)

        # Handle subject ID
        if self.mode in ['eval', 'auto']:
            subj = filedict.get('t1n', filedict.get('t2f', 'unknown'))
        else:
            subj = 'dummy_string'

        return {
            't1n': t1n,
            't1c': t1c,
            't2w': t2w,
            't2f': t2f,
            'missing': missing,
            'subj': subj,
            'filedict': filedict
        }

    def __len__(self):
        return len(self.database)


def clip_and_normalize(img):
    """Clip and normalize image"""
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))
    return img_normalized
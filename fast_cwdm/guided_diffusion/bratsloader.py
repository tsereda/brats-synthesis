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

        # VALIDATED DWT-compatible crop bounds: (160, 208, 160) - all even dimensions
        # These bounds are used everywhere for training, validation, and conversion
        self.crop_bounds = {
            'x_min': 39, 'x_max': 199,  # width: 160
            'y_min': 17, 'y_max': 225,  # height: 208
            'z_min': 0,  'z_max': 160   # depth: 160
        }
        self.cropped_shape = (160, 208, 160)
        print(f"🔧 Using VALIDATED crop: 160x208x160 (DWT-compatible)")

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
        """Apply fixed crop bounds and pad if needed to DWT-compatible shape"""
        x_min, x_max = self.crop_bounds['x_min'], self.crop_bounds['x_max']
        y_min, y_max = self.crop_bounds['y_min'], self.crop_bounds['y_max']
        z_min, z_max = self.crop_bounds['z_min'], self.crop_bounds['z_max']
        cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
        # Pad if needed (shouldn't be needed, but robust)
        pad_shape = self.cropped_shape
        pad_width = [(0, max(0, pad_shape[i] - cropped.shape[i])) for i in range(3)]
        if any(p[1] > 0 for p in pad_width):
            cropped = np.pad(cropped, pad_width, mode='constant')
        return cropped

    def __getitem__(self, x):
        filedict = self.database[x]
        missing = 'none'

        # Helper to process a modality robustly
        def process_modality(key):
            if key in filedict:
                np_img = nibabel.load(filedict[key]).get_fdata()
                cropped = self.apply_crop(np_img)
                normed = clip_and_normalize(cropped)
                tensor = torch.tensor(normed).float().unsqueeze(0)
            else:
                tensor = torch.zeros(1, *self.cropped_shape)
            return tensor

        t1n = process_modality('t1n')
        t1c = process_modality('t1c')
        t2w = process_modality('t2w')
        t2f = process_modality('t2f')

        # Set missing flag if any are missing
        for key, t in zip(['t1n', 't1c', 't2w', 't2f'], [t1n, t1c, t2w, t2f]):
            if t.sum() == 0:
                missing = key

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
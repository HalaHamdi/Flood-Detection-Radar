import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import albumentations as A


def read_data(data_loc, transform, saved=False):
    '''
    This is much like the read_data function in the DataPreperation folder but the purpose is to allow syncing only
    the CNN folder on Google drive to work with Collab instead of the entire directory as in this case, uploading
    the dataset and save files becomes unavoidable.
    '''
    module_dir = os.path.dirname(__file__)
    save_path = os.path.join(module_dir, '../../Saved/ModelPipelines/CNN/data.npy')
    flooded_path = os.path.join(module_dir, f'{data_loc}/flooded/*.jpg')
    non_flooded_path = os.path.join(module_dir, f'{data_loc}/non-flooded/*.jpg')

    if saved:
        try:
            with open(save_path, 'rb') as f:
                x_data = np.load(f, allow_pickle=True)
                y_data = np.load(f, allow_pickle=True)
        except:
            pass
        
    else:
        x_data, y_data= [], []
        
        #load flooded images
        for filename in tqdm(sorted(glob.glob(flooded_path))):
            try:
                 # read in RGB
                img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
                img_p = transform(image=img)['image']
                img_p = np.transpose(img_p, (2, 0, 1))
                x_data.append(img_p)
                y_data.append(1)
            except Exception as e:
               print(str(e))
            
        ## load non-flooded images
        for filename in tqdm(sorted(glob.glob(non_flooded_path))):
            try:
                img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
                img_p = transform(image=img)['image']
                img_p = np.transpose(img_p, (2, 0, 1))
                x_data.append(img_p)
                y_data.append(0)
            except Exception as e:
               print(str(e))

        x_data,  y_data = np.array(x_data), np.array(y_data)
        
        try:
            with open(save_path, 'wb') as f:
                np.save(f, x_data, allow_pickle=True)
                np.save(f, y_data, allow_pickle=True)
        except:
            pass
    
    return x_data, y_data

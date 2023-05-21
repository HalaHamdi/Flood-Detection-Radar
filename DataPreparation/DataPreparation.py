import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import albumentations as A

# check if we are on collab
try:
    import google.colab
    data_loc = '../../../../MyDrive/SI-Project/'
except:
    data_loc = '../../' 

def read_data(gray=False, new_size= 256, normalize=False, transpose=False, more_transforms=None, saved=False):
    '''
    reads the dataset from the folder and returns the train and validation sets.
    Still needs to handle reading the test set.
    Parameters:
        gray: whether to read the images in grayscale or not
        new_size: the size to resize the images to. Helpful for preventing RAM explosion
        normalize: whether to standarize the images or not
        transpose: whether to transpose the images channels for deep learning models or not
        more_transforms: a list of albumentations transforms to apply to the images
        saved: whether to load the data from the saved file or not
    '''
    val_size = 0.2
    module_dir = os.path.dirname(__file__)
    save_path = os.path.join(module_dir, f'{data_loc}Saved/read-data.npy')
    flooded_path = os.path.join(module_dir, f'{data_loc}DataFiles/flooded/*.jpg')
    non_flooded_path = os.path.join(module_dir, f'{data_loc}DataFiles/non-flooded/*.jpg')

    if saved:
        with open(save_path, 'rb') as f:
            x_data = np.load(f, allow_pickle=True)
            y_data = np.load(f, allow_pickle=True)
        
    else:
        x_data, y_data= [], []
        
        #load flooded images
        for filename in tqdm(sorted(glob.glob(flooded_path))):
            try:
                 # read in RGB
                img = cv2.imread(filename, 0 if gray else cv2.COLOR_BGR2RGB)
                img_p = preprocess_img(img, new_size, normalize, more_transforms)
                if transpose: img_p = img_p.transpose(2,0,1)
                x_data.append(img_p)
                y_data.append(1)
            except Exception as e:
               print(str(e))
            
        ## load non-flooded images
        for filename in tqdm(sorted(glob.glob(non_flooded_path))):
            try:
                img = cv2.imread(filename, 0 if gray else cv2.COLOR_BGR2RGB)
                img_p = preprocess_img(img, new_size, normalize,  more_transforms)
                if transpose: img_p = img_p.transpose(2,0,1)
                x_data.append(img_p)
                y_data.append(0)
            except Exception as e:
               print(str(e))

        x_data,  y_data = np.array(x_data), np.array(y_data)
        
        with open(save_path, 'wb') as f:
            np.save(f, x_data, allow_pickle=True)
            np.save(f, y_data, allow_pickle=True)

    # Split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=val_size, random_state=42)
    
    # print x_train and x_val shapes
    print("x_train shape: ", x_train.shape)
    print("x_val shape: ", x_val.shape)
    
    return x_train, x_val, y_train, y_val



def preprocess_img(img, new_size=256, normalize=True, more_transforms=None):
    '''
    Preprocess a given image
    '''
    # Let's make the greater dimension of the image become 256 then apply a center crop of 256x256
    transforms_list = [
        A.SmallestMaxSize(max_size=new_size),
        A.CenterCrop(new_size, new_size),
    ]
    if normalize: transforms_list.append(A.Normalize(mean=[0.277, 0.277, 0.277], std=[0.254, 0.246, 0.223]))
    if more_transforms: transforms_list.extend(more_transforms)
    transform = A.Compose(transforms_list)
    img_p = transform(image=img)['image']
    
    return img_p


def read_sample(sample_path, gray=False, new_size= 256, normalize=False, transpose=False, more_transforms=None):
    ''''
    Reads a single image and make it into an array
    '''
    img = cv2.imread(sample_path, 0 if gray else cv2.COLOR_BGR2RGB)
    img_p = preprocess_img(img, new_size, normalize, more_transforms)
    if transpose: img_p = img_p.transpose(2,0,1)
    img_arr = np.array([img_p])
    return img_arr
    
    

def visualize_data(x_data, y_data, width, height):
    """
    show a grid length num_images of images from x_data with their corresponding labels from y_data
    """
    fig, ax = plt.subplots(height, width, figsize=(width*2, height*2))
    for i in range(height):
        for j in range(width):
            ax[i, j].imshow(x_data[i*width + j])
            ax[i, j].set_title('Flooded' if y_data[i*width + j] else 'Non-Flooded')
            ax[i, j].axis('off')
            
    plt.show()
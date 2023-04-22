import numpy as np
import os

try:
    import google.colab
    data_loc = '../../../../../MyDrive/SI-Project/'
except:
    data_loc = '../../../' 

def color_histogram(img, n_bins):
    '''
    Produces a histogram feature vector for the image. It's of length n_bins if the image is grayscale else 3*n_bins.
    '''
    if len(img.shape) == 2:                         # grayscale
        return np.histogram(img, bins=n_bins, range=(0, 256))[0]
    elif len(img.shape) != 0:
        img_red, img_green, img_blue = img[:,:,0], img[:,:,1], img[:,:,2]
        hist_red = np.histogram(img_red, bins=n_bins, range=(0, 256))[0]
        hist_green = np.histogram(img_green, bins=n_bins, range=(0, 256))[0]
        hist_blue = np.histogram(img_blue, bins=n_bins, range=(0, 256))[0]
        return np.concatenate((hist_red, hist_green, hist_blue))
    else:
        print("ouch")
        return np.array([])



def save_features(apply_rand):
    ''' A decorator to the apply_rand function that extends it so that:
    - it can take x_train and x_val simultaneously
    - it saves the features
    - it can still only take x_test when eval=True
    '''
    def apply_rand_d(x_train, x_val, n_bins, saved=False, eval=False):
        if eval: return apply_rand(x_val, n_bins)
        module_dir = os.path.dirname(__file__)
        save_path = os.path.join(module_dir, f'{data_loc}Saved/hist.npy')
        if saved:    
            with open(save_path, 'rb') as f:
                x_train = np.load(f, allow_pickle=True)
                x_val = np.load(f, allow_pickle=True)
            return x_train, x_val
        
        else:
            x_train = apply_rand(x_train, n_bins)
            x_val = apply_rand(x_val, n_bins)
            
            with open(save_path, 'wb') as f:
                np.save(f, x_train, allow_pickle=True)
                np.save(f, x_val, allow_pickle=True)
            return x_train, x_val
    return apply_rand_d


@save_features
def apply_hist(x_data, n_bins=20):
    '''
    Simply applies the rand function to each image in x_data
    '''
    x_data = np.array([color_histogram(x, n_bins) for x in x_data if x.shape])
    return x_data


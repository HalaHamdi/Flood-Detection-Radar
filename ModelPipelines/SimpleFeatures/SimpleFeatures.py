from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.decomposition import PCA
import numpy as np



def get_lbp(X,radius = 3):
    n_points = 8 * radius
    images=np.array([local_binary_pattern(x, n_points, radius) for x in X])
    images=images.reshape(images.shape[0],images.shape[1]*images.shape[2])
    return images


def get_hog(X,orientation=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3),block_norm='L2-Hys'):
    images=np.array([hog(x,orientations=orientation, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, 
                         block_norm=block_norm, visualize=False, transform_sqrt=False,feature_vector=True)   for x in X])
    return images



def apply_pca(X,n_components=128,pca_obj=None):

    # fit the PCA instance to your data
    if pca_obj is None:
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca, pca.transform(X)
    
    return pca_obj.transform(X)
   
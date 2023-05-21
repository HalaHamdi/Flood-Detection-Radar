# Perform PCA on the features
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.manifold import TSNE
import umap.umap_ as umap
from IPython.display import Image
import imageio
import os

# check if we are on collab
try:
    import google.colab
    data_loc = '../../../../../MyDrive/SI-Project/'
except:
    data_loc = '../../../' 

class illustrate3DFeatures():
    '''
    This class allows visualization of high dimensional features to assess their quality in terms of seperability
    and distribution.
    '''
    def __init__(self, x_data, y_data, feature_name):
        '''
        The init method takes a high dimensional dataset and reduces it into 3D using PCA and UMAP techniques
        '''
        self.x_data = x_data
        self.y_data = y_data
        self.x_data_pca = self.perform_pca()
        self.x_data_umap = self.perform_umap()
        self.feat_name = feature_name
    
    def perform_pca(self, verbose=False):
        '''
        Applies PCA on the high dimensional data and returns the reduced data
        '''
        pca = PCA(n_components=3)
        x_data_r = pca.fit_transform(self.x_data)

        if verbose:
            print(f"Was able to perserve {sum(pca.explained_variance_ratio_[:3])*100}% of the variance")

        return x_data_r
    

    def perform_umap(self, verbose=False):
        '''
        Applies UMAP on the high dimensional data and returns the reduced data.
        UMAP is a non-linear dimensionality reduction technique that (is claimed by the authors) to improve upon t-SNE
        that performs dimensionality reduction for visualization purposes and attempts to show nonlinear seperability if
        it exists unlike PCA.
        '''
        reducer = umap.UMAP(n_components=3)
        x_data_r = reducer.fit_transform(self.x_data)
        return x_data_r
        

    def illustrate_features_3D(self, dim_reduce='PCA', animated=False, show=True):
        '''
        Show a static of an animated plot using either PCA or UMAP as specified in dim_reduce
        '''
        if dim_reduce == 'PCA':
            x_data_r = self.x_data_pca
        elif dim_reduce == 'UMAP':
            x_data_r = self.x_data_umap
        
        plt.style.use('dark_background')                
        fig = plt.figure()
        fig.set_dpi(200)                                                        # increase resolution
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)                                                          # remove grid
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False    # remove principal planes
        # color should be teal or pink depending on class
        colors = np.where(self.y_data==1, 'teal', 'pink')
        ax.scatter(x_data_r[:,0], x_data_r[:,1], x_data_r[:,2], c=colors)
        ax.set_axis_off()                                                       # remove axes
        # add a title
        ax.set_title(f'3D {dim_reduce} dimensionality reduction of {self.feat_name} features', fontsize=10)
        if not animated and show:
            plt.show()
        else:
            frames = []
            num_frames = 72
            for i in range(num_frames):
                ax.view_init(azim=i*5)
                # generate the figure in memory without showing
                fig.canvas.draw()                     
                # convert the saved figure to a 1D numpy array                             
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)   
                # ... 3D numpy array (width, height)->(height, width)->(height, width, 3)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
                # append it into the list of frames that make up the gif
                frames.append(frame)                                               
            path = f'{data_loc}Saved/{self.feat_name}_pca.gif' if dim_reduce == 'PCA' else f'{data_loc}Saved/{self.feat_name}_umap.gif'
            imageio.mimsave(path, frames, duration=1000/15)          # make a gif out of the frames where there are 15 frames per second
            plt.close()
            if show:    display(Image(path))
    
    def double_gif(self, animated=True, useOld=False):
        '''
        Create a gif that shows both PCA and UMAP dimensionality reduction techniques by:
        1 - Checking if the gifs already exist or if they need to be created
        2 - Reading the two gifs and converting to numpy arrays
        3 - Concatenating them along the width axis (horizontally)
        4 - Saving and displaying the gif
        '''
        if not os.path.exists(f'{data_loc}Saved/{self.feat_name}_umap.gif') or not useOld:
            self.illustrate_features_3D(dim_reduce='UMAP', animated=True, show=False)
        if not os.path.exists(f'{data_loc}Saved/{self.feat_name}_pca.gif') or not useOld:
            self.illustrate_features_3D(dim_reduce='PCA', animated=True, show=False)
        gif1 = imageio.mimread(f'{data_loc}Saved/{self.feat_name}_pca.gif', memtest=False)
        gif2 = imageio.mimread(f'{data_loc}Saved/{self.feat_name}_umap.gif', memtest=False)
        gif1 = np.array(gif1)
        gif2 = np.array(gif2)
        gif = np.concatenate((gif1, gif2), axis=2)
        imageio.mimsave(f'{data_loc}Saved/{self.feat_name}_pca-umap.gif', gif, duration=1000/15 if animated else 1, loop=0)
        display(Image(f'{data_loc}Saved/{self.feat_name}_pca-umap.gif'))

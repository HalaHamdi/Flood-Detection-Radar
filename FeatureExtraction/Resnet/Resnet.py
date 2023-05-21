import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import TensorDataset
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.callbacks import EarlyStopping
from torchview import draw_graph
import torchvision.models as models
import numpy as np
import os
from tqdm import tqdm

### collab setup
try:
    import google.colab
    data_loc = '../../../../../MyDrive/SI-Project/'
except:
    data_loc = '../../../' 

    

class ConvNet(pl.LightningModule):
    '''
    Resnet Model
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = models.resnet50(weights="DEFAULT")
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
    def forward(self, x):
        '''
        Forward method to pass a given input batch to the model and return its predictions
        '''
        x = self.model(x.unsqueeze(0))
        return x


def deep_features(img):
    '''
    Produces a feature vector for the image using a pretrained model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    model = ConvNet().to(device)
    model.eval()
    feat = model(img)
    return feat.detach().cpu().numpy()



def save_features(apply_deep_features):
    ''' A decorator to the apply_rand function that extends it so that:
    - it can take x_train and x_val simultaneously
    - it saves the features
    - it can still only take x_test when eval=True
    '''
    def apply_deep_d(x_train, x_val, saved=False, eval=False):
        if eval: return apply_deep_features(x_val)
        module_dir = os.path.dirname(__file__)
        save_path = os.path.join(module_dir, f'{data_loc}Saved/resnet.npy')
        if saved:    
            with open(save_path, 'rb') as f:
                x_train = np.load(f, allow_pickle=True)
                x_val = np.load(f, allow_pickle=True)
            return x_train, x_val
        
        else:
            x_train = apply_deep_features(x_train)
            x_val = apply_deep_features(x_val)
            
            with open(save_path, 'wb') as f:
                np.save(f, x_train, allow_pickle=True)
                np.save(f, x_val, allow_pickle=True)
            return x_train, x_val
    return apply_deep_d


@save_features
def apply_deep_features(x_data):
    '''
    Simply applies the deep_feature function to each image in x_data
    '''
    x_data_f = []
    for x in tqdm(x_data):
        feat = deep_features(x)
        feat = np.squeeze(feat)
        x_data_f.append(feat)
    return np.array(x_data_f)



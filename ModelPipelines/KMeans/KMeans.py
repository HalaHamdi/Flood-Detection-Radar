import sys; sys.path.append('../../')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DataPreparation.DataPreparation import read_data
from sklearn.cluster import KMeans


def get_img(index, x_train_d):
  return np.float32(x_train_d[index].copy()/255)

def show_img(img):
  plt.imshow(img)
  plt.show()

def preprocess_img(img):
  avg_color = []
  for i in range(8):
    avg_color.append(np.mean(img[i*32:(i+1)*32,i*32:(i+1)*32], axis=(0,1)))
  sum = np.sum(avg_color[i][0] > avg_color[i][2] for i in range(8))
  if sum > 4:
    img = img[:,:,::-1]
  return img

def cluster(img):
  if len(img.shape) == 3:
    img_vec = img.reshape(-1,3)
  else:
    img_vec = img.reshape(-1,1)
  img_vec = np.float32(img_vec)
  kmeans = KMeans(n_clusters = 8, n_init=10, random_state=0).fit(img_vec)
  return kmeans

def get_mask(img, kmeans):
  labels = kmeans.labels_
  labels = labels.reshape((img.shape[0],img.shape[1]))
  # Color the labels with cluster centers
  mask = np.zeros_like(img)
  for i in range(len(kmeans.cluster_centers_)):
    mask[labels == i] = kmeans.cluster_centers_[i]
  return mask

def water_mask(img, kmeans):
  labels = kmeans.labels_
  labels = labels.reshape((img.shape[0],img.shape[1]))
  mask = np.zeros_like(img)
  for i in range(len(kmeans.cluster_centers_)):
    hsv = cv2.cvtColor(np.uint8([[kmeans.cluster_centers_[i]*255]]), cv2.COLOR_RGB2HSV)
    hue = hsv[0][0][0]
    if hue > 196-110 and hue < 196+110:
      mask[labels == i] = [0,0,255]
    else:
      mask[labels == i] = [255,255,255]
  return mask

def show_case(img):
  img_proc = preprocess_img(img)
  kmeans = cluster(img_proc)
  water = water_mask(img_proc, kmeans)
  mask = get_mask(img_proc, kmeans)
  fig, ax = plt.subplots(1,4, figsize=(15,15))
  ax[0].imshow(img)
  ax[1].imshow(img_proc)
  ax[2].imshow(water)
  ax[3].imshow(mask)
  plt.show()
  
import albumentations as A
def save_water(sample_path):
  img_p = cv2.imread(sample_path, cv2.COLOR_BGR2RGB)
  img_p = np.array(img_p)
  # center crop to 256x256
  new_size = 256
  transforms_list = [
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(new_size, new_size),
    ]
  transform = A.Compose(transforms_list)
  img_p = transform(image=img_p)['image']

  img_proc = preprocess_img(img_p)
  kmeans = cluster(img_proc)
  water = water_mask(img_proc, kmeans)
  
  # concatenate the original image with the water mask
  water = np.concatenate((img_p, water), axis=1)
  
  # save image bgr to rgb
  water = cv2.cvtColor(water, cv2.COLOR_BGR2RGB)  
  cv2.imwrite('./static/water.jpg', water)
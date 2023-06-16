
# 🌊 Flood Detection and Localization 🚁
<p align="justify"> 
The aim of this project is to utilize computer vision for the purpose of detecting images involving a flood and localizing the flooded areas within them in an automated fashion using classical machine learning and computer vision techniques. Such system shall be useful if employed on a satellite to provide critical information to emergency response teams such as the areas and degrees of flood.
</p>

## 🚀 Pipeline
We solved the detection problem by employing the following pipeline
<img width="1010" alt="image" src="https://github.com/Halahamdy22/Flood_Detection/assets/49572294/c7b35025-7c74-40c1-bcb4-757a05bf218e">

## 📂 Folder Structure
The following is the implied folder structure:
```
.
├── DataPreparation
│   ├── DataPreparation.ipynb
│   └── DataPreparation.py
├── FeatureExtraction
│   ├── Glcm
│   │   ├── Glcm.ipynb
│   │   └── Glcm.py
│   ├── Hist
│   │   ├── Hist.ipynb
│   │   └── Hist.py
│   ├── Resnet
│   │   ├── Resnet.ipynb
│   │   └── Resnet.py
│   ├── Shufflenet
│   │   ├── Shufflenet.ipynb
│   │   └── Shufflenet.py
│   └── Visuals.py
├── ModelPipelines
│   ├── CNN
│   │   └── CNN.ipynb
│   ├── ISODATA
│   │   ├── ISODATA.ipynb
│   │   └── ISODATA.py
│   ├── KMeans
│   │   ├── KMeans.ipynb
│   │   ├── KMeans.py
│   │   └── water.jpg
│   ├── LogisticRegression
│   │   ├── HOG-LogisticRegression.ipynb
│   │   └── LBP-LogisticRegression.ipynb
│   ├── QDA
│   │   ├── Hist-QDA.ipynb
│   │   └── LBP-QDA.ipynb
│   ├── SVM
│   │   ├── Deep-SVM.ipynb
│   │   ├── Glcm-SVM.ipynb
│   │   ├── HOG-SVM.ipynb
│   │   ├── Hist-SVM.ipynb
│   │   ├── LBP-SVM.ipynb
│   │   └── Shuffle-SVM.ipynb
│   └── SimpleFeatures
│       └── SimpleFeatures.py
├── Production
│   └── Flask
├── Quests
├── README.md
├── References
│   ├── Flood-Paper.pdf
│   ├── Relatively Meh Paper.pdf
│   └── SI Project.pdf
├── requirements.txt
└── script.ipynb
```
## 🚁 Running the Project

```python
pip install requirements.txt
# To run any stage of the pipeline, consider the stage's folder. There will always be a demonstration notebook.
```
The project is also fully equipped to run on Google collab if the project's folder is synchronized with your Google drive. You may reach out to any of the developers if you find any difficulty in running it there. 

## 📜 Standards
We have set the following set of working [standards](https://github.com/Halahamdy22/Flood_Detection/tree/main/README-MLDIR.md/) as we were undertaking the project. If you wish to contribute for any reason then please respect such standards.

<hr>

We shall illustrate the pipeline in the rest of the README. For an extensive overview of the project you may choose also checking the [report](https://github.com/Halahamdy22/Flood_Detection/tree/main/Report.pdf/) and the [slides](https://github.com/Halahamdy22/Flood_Detection/tree/main/Presentation.pdf/) or the demonstration notebooks herein.

## 🍳 Data Preparation
![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/98ad4d32-341f-4523-9efb-280463c792a8)

The given dataset involves 922 images equally divided into flooded and non-flooded images that vary significantly in terms of the quality, size and content. We employed a data processing stage with the following capabilities:
- Reading the images
- Image Standardization
- Greyscale Conversion
- Center-cropping with Maximal Content
- Swapping Channels
- Tensor Conversions and Custom Processing
- Saving the preprocessed images

## 🌟 Feature Extraction & Analysis
We have considered the following set of features
|     GLCM     | ColorHistogram |        Histogram of Gradients     |  Local Binary Pattern   | ResNet      | ShuffleNet  |
|--------------|----------------|-----------------------------------|-------------------------|-------------|-------------|

where in all cases PCA was also an option.

### 🌇 GLCM Features
GLCM features are just statistics acquired from the gray-level co-occurrence matrix of the image. The following analyzes the target's separability under all possible pairs of GLCM features
![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/713da7dc-83e9-477f-9b00-7de8704f26f3)

To illustrate all of them together, we have utilized linear (PCA) and non-linear (UMAP) dimensioanlity reduction

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/de6cf3a3-3bc3-41c6-afa1-a76e82c05eed)

### 🎨 Color Histogram

In this, we simply sampled the color histogram to form a feature vector for the image. Surprisingly better than expected.

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/e986e36d-8e7c-4c31-b34c-786c8231d1aa)

### 📏 Histogram of Gradients

In this, we simply described the image by a histogram of gradients; this is useful since edges carry important info about whether the image contains a flood or not. 

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/c3454be2-340d-41ca-a61e-4dd0f46bf682)

### 📱 Local Binary Pattern

A sliding window over the image is used to detect patterns between the center pixel and the rest of the pixels; hence, detecting repeating patterns. This is helpful since flooded images will often lack texture compared to non-flooded ones.

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/a87f20e6-7059-42cd-863d-d8194110cae6)

### 🦑 ResNet Features

Eventually, we decided to try features extracted by deep learning computer vision models. Our first choice was the ResNet-50 CNN model.

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/dd5435ae-1f08-4b0d-9c9b-2f3628aeb9dd)

Notice that it is no longer fruit salad.

### 🃏 ShuffleNet Features

After a thorough comparison of various transfer learning options, we found it opportune to try out ShuffleNet features - a light-weight model with performance that matches or exceeds much larger models such as ResNet-50.

![output](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/3fb7bfde-4371-4f1f-8fa8-ac6f897e6ab6)


## 🚢 Model Building
<table style="width: 100%; text-align: center;">
  <tr>
    <th style="width: 25%;">Model</th>
    <th style="width: 75%;" colspan="6">Features</th>
  </tr>
  <tr>
    <td style="width: 25%;">CNN</td>
    <td style="width: 75%;" colspan="6">Raw Images</td>
  </tr>
  <tr>
    <td style="width: 25%;">Logistic Regression</td>
    <td style="width: 12.5%;" colspan="2">Hog</td>
    <td style="width: 12.5%;" colspan="4">LBP</td>
  </tr>
  <tr>
    <td style="width: 25%;">Bayes Classifier</td>
    <td style="width: 12.5%;" colspan="2">ColorHist</td>
    <td style="width: 12.5%;" colspan="4">LBP</td>
  </tr>
  <tr>
    <td style="width: 25%;">SVM</td>
    <td style="width: 12.5%;">ResNet</td>
    <td style="width: 12.5%;">GLCM</td>
    <td style="width: 12.5%;">ColorHist</td>
    <td style="width: 12.5%;">HoG</td>
    <td style="width: 12.5%;">LBP</td>
    <td style="width: 12.5%;">ShuffleNet</td>
  </tr>
</table>


### CNN

We built a CNN using Pytorch Lightning, converging with LeNet after a lot of hyperparameter tuning. Performance was not overly spectacular.

![output (1)](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/fe45ee52-3de1-43b0-b8e8-9c7578bfb2ce)

For the rest of the pipelines, we used extracted features as shown above. We also used in-notebook logging using the <a href="https://github.com/EssamWisam/MLPath"> MLPath library </a> as present in the demonstration notebooks and the [report](https://github.com/Halahamdy22/Flood_Detection/tree/main/Report.pdf/). The following shows a sample of the log for the LBP-Logistic pipeline:

<table>
<tr>
<th colspan=4 style="text-align: center; vertical-align: middle;">info</th>
<th colspan=5 style="text-align: center; vertical-align: middle;">read_data</th>
<th colspan=1 style="text-align: center; vertical-align: middle;">get_lbp</th>
<th colspan=2 style="text-align: center; vertical-align: middle;">apply_pca</th>
<th colspan=11 style="text-align: center; vertical-align: middle;">LogisticRegression</th>
<th colspan=2 style="text-align: center; vertical-align: middle;">metrics</th>
</tr>
<th style="text-align: center; vertical-align: middle;">time</th>
<th style="text-align: center; vertical-align: middle;">date</th>
<th style="text-align: center; vertical-align: middle;">duration</th>
<th style="text-align: center; vertical-align: middle;">id</th>
<th style="text-align: center; vertical-align: middle;">gray</th>
<th style="text-align: center; vertical-align: middle;">saved</th>
<th style="text-align: center; vertical-align: middle;">new_size</th>
<th style="text-align: center; vertical-align: middle;">normalize</th>
<th style="text-align: center; vertical-align: middle;">transpose</th>
<th style="text-align: center; vertical-align: middle;">radius</th>
<th style="text-align: center; vertical-align: middle;">n_components</th>
<th style="text-align: center; vertical-align: middle;">pca_obj</th>
<th style="text-align: center; vertical-align: middle;">penalty</th>
<th style="text-align: center; vertical-align: middle;">C</th>
<th style="text-align: center; vertical-align: middle;">dual</th>
<th style="text-align: center; vertical-align: middle;">tol</th>
<th style="text-align: center; vertical-align: middle;">fit_intercept</th>
<th style="text-align: center; vertical-align: middle;">intercept_scaling</th>
<th style="text-align: center; vertical-align: middle;">solver</th>
<th style="text-align: center; vertical-align: middle;">max_iter</th>
<th style="text-align: center; vertical-align: middle;">multi_class</th>
<th style="text-align: center; vertical-align: middle;">verbose</th>
<th style="text-align: center; vertical-align: middle;">warm_start</th>
<th style="text-align: center; vertical-align: middle;">Accuracy</th>
<th style="text-align: center; vertical-align: middle;">F1</th>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;"> <font color=white>14:29:58</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>05/20/23</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1.10 min</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>256</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>5</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>512</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>PCA(n_components=512)</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>l2</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>7</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0.0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>lbfgs</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>100</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>auto</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0.5891891891891892</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0.5313333333333333</font></td>
</tr>
<tr>
<td style="text-align: center; vertical-align: middle;"> <font color=white>14:31:55</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>05/20/23</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>34.83 s</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>2</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>256</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>5</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow></font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow></font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>l2</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>7</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0.0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>lbfgs</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>100</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>auto</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>0.5837837837837838</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>0.5633716475095785</font></td>
</tr>

<tr>
<td style="text-align: center; vertical-align: middle;"> <font color=white>14:38:15</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>05/20/23</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1.73 min</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>8</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>256</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>4</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>512</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>PCA(n_components=512)</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>l2</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>100</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0.0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>True</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>1</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>lbfgs</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>100</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>auto</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>0</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=white>False</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>0.7567567567567568</font></td>
<td style="text-align: center; vertical-align: middle;"> <font color=yellow>0.756043956043956</font></td>
</tr>
</table>

Note that hyperparameter search was also used for high-performing models such as SVM.

## Metrics & Results 📉
For evaluation, we used a 20% validation set split and 10-Repeated-5-Fold Cross Validation for more fine-grained comparisons (e.g., ShuffleNet VS. ResNet). The following portrays the results:

| Method          | HOG-LR | LBP-LR | GLCM-SVM | HOG-SVM | LBP-SVM | Shufflenet-SVM | Resnet-SVM | Hist-QDA | LBP-QDA |
|-----------------|--------|--------|----------|---------|---------|----------------|------------|----------|---------|
| F1-score        | 0.854  | 0.756  | 0.683    | 0.897   | 0.832   | 0.989          | 0.979      | 0.816    | 0.619   |

Overall, ShuffleNet was our best model. In light of manual error analysis, there were its mistakes on the validation set

![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/f78a8030-2fad-4ae2-b71a-fef833a4f037)


## 🩺 Retrospective Analysis

After submitting ShuffleNet to the competition, we found that it has ranked only 4th place relative to the other accuracies in the leaderboard with an accuracy of 96.5%. We requested the test set to do some retrospective analysis and found out that ResNet actually performs significantly better on the test set at 98% (contrary to performance on the original dataset where ShuffleNet had 98% and ResNet had 97.7% under 10-repeated-5-Fold cross validation) and that ShuffleNet has made mistakes for the following

![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/c74ce3a1-3700-42d2-a152-d2dc2c39cf76)

After revisiting 10-Repeated-5-Fold cross validation for both ResNet and ShuffleNet we concluded that over the 50 random validation splits, the standard deviation was as large as 1.1% for ResNet and 1% for ShuffleNet which is even further aggravated for 10-Repeated-10-Fold cross validation. We could only explain this unexpectedly high sensitivity to the split by the large variance inherent in the data itself as it seems to be collected from different sources. This signals that decisions taken by individuals under a fixed validation set may be really sensitive on how it tallies with the actual test set which is randomly decided; the STD is high compared to a typical dataset. 

#### We are halfway there. The corrolary aspect of the problem is to localize the flooded pixels

###  💦 Flood Segmentation
For this task, we considered ISODATA and K-Means to segment the flooded images. K-Means proved more successful on that front. 

Two obstacles in this task were red water and luminance effects our approach to circumventing them involved swapping channels and utiliting HSV channels respectivelty.

#### 💧 K-Means Results

Classic Example

![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/074896ff-8e11-4fa7-8c9b-ad11939b0c10)

Red Water

![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/60553c3d-251b-41b8-a35f-19a44b39ebae)

Luminance Problems

![image](https://github.com/Halahamdy22/Flood_Detection/assets/49572294/be312392-39e8-4534-93d7-faf05dfe5a23)

Such masterpiece surely deserved a
#### 🌐 Web Interface

```python
cd Production/Flask
flask run
```

<img width="1349" alt="image" src="https://github.com/Halahamdy22/Flood_Detection/assets/49572294/c4144ceb-d8bb-45da-8dfb-9b43ebe7e3f8">

<img width="1349" alt="image" src="https://github.com/Halahamdy22/Flood_Detection/assets/49572294/603237af-0519-4ae2-9b2d-8bb573f2583a">


## Collaborators

<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/EssamWisam">
            <img src="https://avatars.githubusercontent.com/u/49572294?v=4" width="100;" alt="EssamWisam"/>
            <br />
            <sub><b>Essam</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Muhammad-saad-2000">
            <img src="https://avatars.githubusercontent.com/u/61880555?v=4" width="100;" alt="Muhammad-saad-2000"/>
            <br />
            <sub><b>MUHAMMAD SAAD</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/NouranHany">
            <img src="https://avatars.githubusercontent.com/u/59095993?v=4" width="100;" alt="NouranHany"/>
            <br />
            <sub><b>Noran Hany</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Halahamdy22">
            <img src="https://avatars.githubusercontent.com/u/56937106?v=4" width="100;" alt="Halahamdy22"/>
            <br />
            <sub><b>Halahamdy22</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->


# 🌊 Flood Detection and Localization 🚁
<p align="justify"> 
The aim of this project is to utilize computer vision for the purpose of detecting images involving a flood and localizing the flooded areas within them in an automated fashion using classical machine learning and computer vision techniques. Such system shall be useful if employed on a satellite to provide critical information to emergency response teams such as the areas and degrees of flood.
</p>

## 🚀 Pipeline
We solved the detection problem by employing the following pipeline
<img width="1010" alt="image" src="https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246481747-c7b35025-7c74-40c1-bcb4-757a05bf218e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143031Z&X-Amz-Expires=300&X-Amz-Signature=8cc15f31a5a5f6338acf8def4f3018f7fbe3648a973e231da55cc190a365625c&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215">

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
![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246491452-98ad4d32-341f-4523-9efb-280463c792a8.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143052Z&X-Amz-Expires=300&X-Amz-Signature=e2bdc2c741c236bc36e8f92d17e3186b962487b2a4f4a0d1384ec1046801d546&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

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
![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246500264-713da7dc-83e9-477f-9b00-7de8704f26f3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143104Z&X-Amz-Expires=300&X-Amz-Signature=f7d21ba4cf34919bfae0b53e0da25ab580b805ee898cfca49dd5bbe7e028785c&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

To illustrate all of them together, we have utilized linear (PCA) and non-linear (UMAP) dimensioanlity reduction

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246503754-de6cf3a3-3bc3-41c6-afa1-a76e82c05eed.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143116Z&X-Amz-Expires=300&X-Amz-Signature=ef1e1dfd69634f7f68e00247ee64c19765995e4063928d2ddda61cde8223d9a2&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

### 🎨 Color Histogram

In this, we simply sampled the color histogram to form a feature vector for the image. Surprisingly better than expected.

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246510914-e986e36d-8e7c-4c31-b34c-786c8231d1aa.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143125Z&X-Amz-Expires=300&X-Amz-Signature=abf2f902b7653a7e7e7bbc7da878072cc42f259c85e154045cabd2b579d568cb&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

### 📏 Histogram of Gradients

In this, we simply described the image by a histogram of gradients; this is useful since edges carry important info about whether the image contains a flood or not. 

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246519324-c3454be2-340d-41ca-a61e-4dd0f46bf682.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143136Z&X-Amz-Expires=300&X-Amz-Signature=8f1e20b14afbe2997e39a8a4e67a49c0b42c21f1addae25f402f2bd04469fd10&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

### 📱 Local Binary Pattern

A sliding window over the image is used to detect patterns between the center pixel and the rest of the pixels; hence, detecting repeating patterns. This is helpful since flooded images will often lack texture compared to non-flooded ones.

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246520665-a87f20e6-7059-42cd-863d-d8194110cae6.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143147Z&X-Amz-Expires=300&X-Amz-Signature=b3d8cafc9527e1fdd046e9ab25d3455424c1fca0c10ca2f118eccfd149873aaa&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

### 🦑 ResNet Features

Eventually, we decided to try features extracted by deep learning computer vision models. Our first choice was the ResNet-50 CNN model.

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246522319-dd5435ae-1f08-4b0d-9c9b-2f3628aeb9dd.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143204Z&X-Amz-Expires=300&X-Amz-Signature=8e4f83db8a2743106ede8bb0c2ca2dfe7b0f965ebff6086f0393625bd9a7c0f6&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

Notice that it is no longer fruit salad.

### 🃏 ShuffleNet Features

After a thorough comparison of various transfer learning options, we found it opportune to try out ShuffleNet features - a light-weight model with performance that matches or exceeds much larger models such as ResNet-50.

![output](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246523178-3fb7bfde-4371-4f1f-8fa8-ac6f897e6ab6.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143214Z&X-Amz-Expires=300&X-Amz-Signature=5550a29f548cfe13d868eca2b035c29b433edfe8d1b46bc03c8aaac468a754eb&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)


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

![output (1)](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246525796-fe45ee52-3de1-43b0-b8e8-9c7578bfb2ce.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143225Z&X-Amz-Expires=300&X-Amz-Signature=9880ab4ba579e6de2cb9bc332e1cdf2473be70c3fd0a684999ac28a23aa96a61&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

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

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246532557-f78a8030-2fad-4ae2-b71a-fef833a4f037.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143244Z&X-Amz-Expires=300&X-Amz-Signature=406579a6b74a265946fa7b462a80b2e848c4a1635ea2e0ff3433b534cdf3bff1&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)


## 🩺 Retrospective Analysis

After submitting ShuffleNet to the competition, we found that it has ranked only 4th place relative to the other accuracies in the leaderboard with an accuracy of 96.5%. We requested the test set to do some retrospective analysis and found out that ResNet actually performs significantly better on the test set at 98% (contrary to performance on the original dataset where ShuffleNet had 98% and ResNet had 97.7% under 10-repeated-5-Fold cross validation) and that ShuffleNet has made mistakes for the following

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246533541-c74ce3a1-3700-42d2-a152-d2dc2c39cf76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143255Z&X-Amz-Expires=300&X-Amz-Signature=e4c961a1424a14812d3c5992f09a689892509dfd4cdec655076047f49fd27cfa&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

After revisiting 10-Repeated-5-Fold cross validation for both ResNet and ShuffleNet we concluded that over the 50 random validation splits, the standard deviation was as large as 1.1% for ResNet and 1% for ShuffleNet which is even further aggravated for 10-Repeated-10-Fold cross validation. We could only explain this unexpectedly high sensitivity to the split by the large variance inherent in the data itself as it seems to be collected from different sources. This signals that decisions taken by individuals under a fixed validation set may be really sensitive on how it tallies with the actual test set which is randomly decided; the STD is high compared to a typical dataset. 

#### We are halfway there. The corrolary aspect of the problem is to localize the flooded pixels

###  💦 Flood Segmentation
For this task, we considered ISODATA and K-Means to segment the flooded images. K-Means proved more successful on that front. 

Two obstacles in this task were red water and luminance effects our approach to circumventing them involved swapping channels and utiliting HSV channels respectivelty.

#### 💧 K-Means Results

Classic Example

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246542086-074896ff-8e11-4fa7-8c9b-ad11939b0c10.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143307Z&X-Amz-Expires=300&X-Amz-Signature=7cdab8e6ff083f51f363c93067acb639b7763c687d3067c8179035b6138346d5&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

Red Water

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246542119-60553c3d-251b-41b8-a35f-19a44b39ebae.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143318Z&X-Amz-Expires=300&X-Amz-Signature=1dc17cf400146418e7d9143632ee905744e19b6c8e7a7adff64f5e48d4c910e4&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

Luminance Problems

![image](https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246542183-be312392-39e8-4534-93d7-faf05dfe5a23.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143328Z&X-Amz-Expires=300&X-Amz-Signature=8bf516793ec34f9fdd93b24d2a88a563a1a4625e1e8b91514ad359a1ee71c5a5&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215)

Such masterpiece surely deserved a
#### 🌐 Web Interface

```python
cd Production/Flask
flask run
```

<img width="1349" alt="image" src="https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246545205-c4144ceb-d8bb-45da-8dfb-9b43ebe7e3f8.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143338Z&X-Amz-Expires=300&X-Amz-Signature=2bfad0d657194088e4a4ad7740c5d7f8f17fefe08bcf21d80785f2e8543e46b9&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215">

<img width="1349" alt="image" src="https://github-production-user-asset-6210df.s3.amazonaws.com/49572294/246545772-603237af-0519-4ae2-9b2d-8bb573f2583a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230617T143347Z&X-Amz-Expires=300&X-Amz-Signature=933f6d88834ff950ed3b55b497ee218326725b72ceaa20cdcf7a52334fef43af&X-Amz-SignedHeaders=host&actor_id=49572294&key_id=0&repo_id=625257215">


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

## 📈 Progress Tracking
We have utilized [Notion](https://www.notion.so/) for progress tracking and task assignment among the team.

<h2 align="center"> 💖 Thank you. 💖 </h2>


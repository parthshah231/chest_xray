# Project Task

To classify a normal chest xray to the one with bacterial pneumonia.
> Recorded an F1-score of 98.10%

## Dataset

- Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
- For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
- Link to the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

![Types of Chest xrays in dataset](https://github.com/parthshah231/chest_xray/blob/master/README/for_git.JPG)

- After searching and sorting through notebooks for the following dataset, the best notebook I found recorded the results **Recall: 98%, Precision: 79%**
- My current best model - Reset 34 with subjectwise IntensityRescale records **Recall: 98.72%, Precision: 97.48%**

## Approach

-   Images were not of the same size so implemented [patch based](https://arxiv.org/abs/2201.09792) training. Every epoch the dataloader would sample a random patch (256 x 256) from the x-ray image and train the network based on that.
-   Divided an image into 6 equal parts and cropped the first part from either sides, as that region is mostly going to be pitch black and not going to contribute anything to the training model.
-   Visual inspection of the data revealed that images had quite different mean intensities (exposures). Pneuomonia images also tended to have a brighter, cloudier appearance (presumably due to presence of infection in the lungs), making the overall intensity for these images also higher. This means that over-exposed images spuriously resemble pneumonia images, and it seemed likely that this was confounding the learner. [_RescaleIntensity_](https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.RescaleIntensity) was experimented with to solve this problem, and was extremely successful (accuracy increased from 71% to 92%).
-   Augmentation was necessary to prevent overfitting on this small dataset. [_RandomErasing_](https://arxiv.org/abs/1708.04896) was the one that made most sense with xray data as there is a possiblity that an xray could just have a small patch of black some where and it might be just noise.
-   Augmentation is done on the patches here instead of the whole image because as dataloader samples a random patch there is a possibility that it might miss the erased part. But applying _RandomErasing_ on a patch with a probability factor would give more control over the augmentation factor.

## Results

After conducting an ablation study, the results are recorded below, please note the batch_size for all the models below is 32.


![subjectWise_intensityRescale](https://github.com/parthshah231/chest_xray/blob/master/README/subjectWise_intensityRescale.png)

From the above plot, we can infer: 
- Resnets perform well overall.
- Simple-conv has a higher precision compared to all meaning there were very few false
positives i.e. the deeper models ensure that they don't classify healthy people as pneunomia patients.
- Linear has a very high recall, which tells us that it correctly classifies all the
patients who have pneumonia, irrespective of the fact that it may have classfied
a few people who didn't have pnemonia as pnemonia patients.

![patchWise_intensityRescale](https://github.com/parthshah231/chest_xray/blob/master/README/patchWise_intensityRescale.png)


From the above plot, we can infer: 
- Resnets perform well overall.
- Simple-conv and Linear both struggle with precision, which leads to affecting their
f1-score.

### Model : Resnet - 18

|  Re-scaling  | Recall | Precision | F1 Score | Accuracy |
| :----------: | :----: | --------: | -------: | -------: |
| subject-wise | 0.9788 |    0.9746 |   0.9767 |   0.9632 |
|  patch-wise  | 0.9788 |    0.9625 |   0.9705 |   0.9573 |

### Model : Resnet - 34

|  Re-scaling  | Recall | Precision | F1 Score | Accuracy |
| :----------: | :----: | --------: | -------: | -------: |
| subject-wise | 0.9872 |    0.9748 |   0.9810 |   0.9725 |
|  patch-wise  | 0.9788 |    0.9585 |   0.9685 |   0.9542 |

### Model : SimpleConv
|  Re-scaling  | Recall | Precision | F1 Score | Accuracy |
| :----------: | :----: | --------: | -------: | -------: |
| subject-wise | 0.7711 |    0.9891 |   0.8666 |   0.8292 |
|  patch-wise  | 0.9872 |    0.9505 |   0.9588 |   0.9541 |


### Model : SimpleLinear

|  Re-scaling  | Recall | Precision | F1 Score | Accuracy |
| :----------: | :----: | --------: | -------: | -------: |
| subject-wise | 0.9788 |    0.7598 |   0.8555 |   0.7621 |
|  patch-wise  | 1.0    |    0.7375 |   0.8473 |   0.7408 |


## Futher Implementation
- Implementing and testing the results of TrivialAugment
- Optimizing Convolution model to make it more dynamic
- Implementing other [transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) models such as EfficientNet, MobileNetV2
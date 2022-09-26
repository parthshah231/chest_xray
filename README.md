# Project Task

To classify a normal chest xray to the one with bacterial pneumonia.
> Recorded an accuracy of 98.32%


## Approach

-   Images were not of the same size so implemented [patch based](https://arxiv.org/abs/2201.09792) training. Every epoch the dataloader would sample a random patch (256 x 256) from the x-ray image and train the network based on that.
-   Divided an image into 6 equal parts and cropped the first part from either sides, as that region is mostly going to be pitch black and not going to contribute anything to the training model.
-   Visual inspection of the data revealed that images had quite different mean intensities (exposures). Pneuomonia images also tended to have a brighter, cloudier appearance (presumably due to presence of infection in the lungs), making the overall intensity for these images also higher. This means that over-exposed images spuriously resemble pneumonia images, and it seemed likely that this was confounding the learner. [_RescaleIntensity_](https://torchio.readthedocs.io/transforms/preprocessing.html#torchio.transforms.RescaleIntensity) was experimented with to solve this problem, and was extremely successful (accuracy increased from 71% to 92%).
-   Augmentation was necessary to prevent overfitting on this small dataset. [_RandomErasing_](https://arxiv.org/abs/1708.04896) was the one that made most sense with xray data as there is a possiblity that an xray could just have a small patch of black some where and it might be just noise.
-   Augmentation is done on the patches here instead of the whole image because as dataloader samples a random patch there is a possibility that it might miss the erased part. But applying _RandomErasing_ on a patch with a probability factor would give more control over the augmentation factor.

## Results

### Intensity Rescaling Experiments

|  Re-scaling  | Recall | Precision | F1 Score |
| :----------: | :----: | --------: | -------: |
| subject-wise | 0.9788 |    0.9788 |   0.9788 |
|  patch-wise  | 0.9788 |    0.9665 |   0.9726 |

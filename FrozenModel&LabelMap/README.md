# EfficientNet

I used the EfficientNet V2 from Tensorflow 2 Zoo to train a fish detection model with 1000 images from the OpenImagesV4 Dataset. It has been trained 15000 steps.

The model pipeline was modified to have a batch size of 4, with an Adam Optimizer of learning rate 1 x 10^-4

From the results, it appears that the loss has started to reach its ceiling. Hence i decided to stop training.


## Loss Metrics
<img align = "centre" src = LossMetrics.png>


## Results
<img align = "centre" src = PictureResults.png>
# SmartMov

The aim of this network is to be able to detect and identify moving objects in extreme climatic conditions.

This network is the cascading of two different ones.
To perform the motion masks we have created a convolutional LSTM U-Net. For object prediction we use a Mask-RCNN.

The Mask-RCNN is a network capable of adapting to any type of data, which is not the case with our U-Net, which detects movement in the scene. It is therefore necessary to train this network on your dataset so that it is able to adapt. The training method is detailed below.

# Results

We evaluated our network on different sequences of the CD-NET2014 dataset (skating, PETS2006, pedestrians, blizzard, snowFall, streetCornerAtNight, highway) as well as on a video (Polytech) we annotated ourselves.
The metrics we used are IoU and F1-score. The results are presented in the following table:

| Metrics  | skating | PETS2006 | pedestrians | blizzard | snowFall | streetCorner | highway | Polytech | Mean  |
| :------: | :-----: | :------: | :---------: | :------: | :------: | :----------: | :-----: | :------: | :--:  |
| IoU      | 0.816   | 0.82     | 0.62        | 0.75     | 0.56     | 0.38         | 0.64    | 0.62     | 0.651 |
| F1-score | 0.847   | 0.93     | 0.72        | 0.839    | 0.884    | 0.429        | 0.789   | 0.834    | 0.784 |

The visual results are presented in the following video: https://www.youtube.com/watch?v=WBlZlWDwU8s

# Environment
This code has been developed on Tensorflow 2.1.0 and Windows 10, Python=3.7.6
Other necessary packages : imgaug, opencv-python, Pillow, xlwings, numpy, matplotlib, scikit-image, scikit-learn

# Download pre-trained models
Our pre-trained models (on the datasets shown above) are available at the following address: https://drive.google.com/file/d/1xvM19n1kdsNhRa0gZoagWEjHSH7paDxe/view?usp=sharing

At this address are also present the mask-RCNN models trained on the COCO2014 dataset to predict only people, cars, or both.
The downloaded models must then be placed in the models directory, which itself contains two sub-folders: Mask-RCNN and U-Net.

# Predict
Once the models are loaded, prediction can be made. The predict.py file in the samples folder allows you to make this prediction from the images in the directory specified in the code.

# Training

## Mask-RCNN
The training of the Mask-RCNN is done as presented in the file train_rcnn.py in the samples folder. It is based on the ideas of the following article: https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d

## U-Net
U-Net training is done as presented in the train_unet.py file in the samples folder. It can be done in two ways: by using a DataGenerator object or by simply specifying a folder containing the images and groundtruths to be used for training.

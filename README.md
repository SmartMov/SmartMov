# SmartMov

The aim of this network is to be able to detect and identify moving objects in extreme climatic conditions.

This network is the cascading of two different ones.
To perform the motion masks we have created a convolutional LSTM U-Net. For object prediction we use a Mask-RCNN.

The Mask-RCNN is a network capable of adapting to any type of data, which is not the case with our U-Net, which detects movement in the scene. It is therefore necessary to train this network on your dataset so that it is able to adapt. The training method is detailed below.

# Results

We evaluated our network on different sequences of the CD-NET2014 dataset (skating, PETS2006, pedestrians, blizzard, snowFall, streetCornerAtNight, highway) as well as on a video (Polytech) we annotated ourselves.
The metrics we used are IoU and F1-score. The results are presented in the following table:

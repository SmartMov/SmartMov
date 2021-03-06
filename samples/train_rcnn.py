import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import PIL
import time
import cv2

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from SmartMov import SmartMov
import coco
from utils import sorted_nicely

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    NUM_CLASSES = 1 + 2 # Background + Person + Cars
    NAME = 'coco_person_car'
    STEPS_PER_EPOCH = 2000 # Nombre d'images dans la datset à utiliser pour l'entrainement

config = InferenceConfig()
config.display()

class_names = ['BG','person','car']

s=(480,720,3)

MODELS_DIR = os.path.join(ROOT_DIR,"models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")
MODELS_MASKRCNN_DIR = os.path.join(MODELS_DIR,"Mask-RCNN/")
DATASET_DIR = os.path.join(ROOT_DIR,"../Datasets/")

TIMESTEP = 5

#%% Création du détecteur
smartmov = SmartMov()

#%% Chargement du U-Net
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_highway.h5", shape_unet=s, timestep=TIMESTEP)

#%% Load dataset train

COCO_DIR = os.path.join(DATASET_DIR,"coco2014/")

coco_train = coco.CocoDataset() # Objet qui va contenir les images et les masques d'entrainement
coco_train.load_coco(COCO_DIR, "train", year='2014', auto_download=True, class_ids=[1,3])
coco_train.prepare()

coco_val = coco.CocoDataset() # Objet qui va contenir les images et les masques de validation
cc = coco_val.load_coco(COCO_DIR, "minival", year='2014', auto_download=True, return_coco=True, class_ids=[1,3])
coco_val.prepare()

#%% Création du Mask-RCNN
smartmov.create_model('rcnn',model_dir=MODELS_MASKRCNN_DIR+"logs/",config=config,reuse=True,
                          model_rcnn=MODELS_MASKRCNN_DIR+"mask_rcnn_coco.h5",class_names=class_names)

#%% Entrainement du Mask-RCNN (warnings à ignorer)
smartmov.train('rcnn',dataset_train_rcnn=coco_train,dataset_val_rcnn=coco_val,
                    epochs_rcnn=3, layers_rcnn='heads')

#%% Conversion du Mask-RCNN en mode inférence pour pouvoir faire les prédictions
smartmov.convert_rcnn() # Prends du temps

#%% Sauvegarde du modèle du Mask-RCNN
smartmov.save(models_to_save='rcnn',dir_rcnn=os.path.join(MODELS_MASKRCNN_DIR,"mask_rcnn_example.h5"))

#%% Prédiction

IMAGES_DIR = os.path.join(ROOT_DIR,"dataset_test/input/") # Dossier des images à prédire (doivent correspondre au U-Net chargé avant)

input_list = sorted_nicely(glob.glob(IMAGES_DIR+"*.jpg")) # Liste ordonnée des images à évaluer

im = []
for f in input_list:
    im.append(plt.imread(f)) # Chargement des images
im = np.array(im)

pred = smartmov.predict(im,models_to_use='all') # pred contiendra (prediction globale, nombre d'objets détectés)
pred_unet = smartmov.predict(im,models_to_use='unet') # pred_unet contiendra uniquement un masque binaire
pred_rcnn = smartmov.predict(im,models_to_use='rcnn') # pred_rcnn contiendra la prediction du Mask-RCNN

#%% Visualisation
smartmov.visualize(im[-1],pred)
smartmov.visualize(im[-1],pred_rcnn,models_used='rcnn')
smartmov.visualize(im[-1],pred_unet,models_used='unet')

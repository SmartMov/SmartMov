import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from SmartMov import SmartMov
from utils import sorted_nicely

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    NUM_CLASSES = 1 + 2 # Background + Person + Cars
    NAME = 'coco_person_car'
    STEPS_PER_EPOCH = 2000

config = InferenceConfig()
config.display()

class_names = ['BG','person','car']

s=(480,720,3) # Shape d'entrée du U-Net

MODELS_DIR = os.path.join(ROOT_DIR,"models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")
MODELS_MASKRCNN_DIR = os.path.join(MODELS_DIR,"Mask-RCNN/")
DATASET_DIR = os.path.join(ROOT_DIR,"../Datasets/")

TIMESTEP=5

#%% Création du détecteur
smartmov = SmartMov()

#%% Chargement des deux réseaux
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5',
                     config=config, class_names=class_names)
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_highway.h5", shape_unet=s, timestep=TIMESTEP)

#%% Chargement des images sur lesquelles réaliser la prédiciton

IMAGES_DIR = os.path.join(ROOT_DIR,"dataset_test/input/") # Dossier des images à prédire (doivent correspondre au U-Net chargé avant)

input_list = sorted_nicely(glob.glob(IMAGES_DIR+"*.jpg")) # Liste ordonnée des images à évaluer

im = []
for f in input_list:
    im.append(plt.imread(f)) # Chargement des images
im = np.array(im)

#%% Prédictions en utilisant chacun des trois réseaux

pred = smartmov.predict(im,models_to_use='all', priority='rcnn') # pred contiendra (prediction globale, nombre d'objets détectés)
pred_unet = smartmov.predict(im,models_to_use='unet') # pred_unet contiendra uniquement un masque binaire
pred_rcnn = smartmov.predict(im,models_to_use='rcnn') # pred_rcnn contiendra la prediction du Mask-RCNN

#%% Visualisation des prédictions

smartmov.visualize(im[-1],pred)
smartmov.visualize(im[-1],pred_rcnn,models_used='rcnn')
smartmov.visualize(im[-1],pred_unet,models_used='unet')

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
from utils import barre, to_excel, sorted_nicely, get_new_colors
from metrics import f1_score

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

TIMESTEP = 5

#%% Création du détecteur
smartmov = SmartMov()

#%% Chargement des deux réseaux
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5',
                     config=config, class_names=class_names)
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_highway.h5", shape_unet=s, timestep=TIMESTEP)

#%% Chargement des données sur lesquelles évaluer le modèle

IMAGES_DIR = os.path.join(ROOT_DIR,"dataset_test/input/") # Dossier des images à évaluer (doivent correspondre au U-Net chargé avant)
GT_DIR = os.path.join(ROOT_DIR,"dataset_test/groundtruth/") # Dossier de la groundtruth correspondante à ces images

input_list = sorted_nicely(glob.glob(IMAGES_DIR+"*.jpg")) # Liste ordonnée des images à évaluer
gt_list = sorted_nicely(glob.glob(GT_DIR+"*.png")) # Liste ordonnée des groundtruth

im = []
for f in input_list:
    im.append(plt.imread(f)) # Chargement des images
im = np.array(im)

gt = np.array(PIL.Image.open(gt_list[-1])) # La groundtruth correspondante aux 5 images est la dernière
if gt.dtype!=np.bool:
    gt[(gt<255) & (gt>1)] = 0
    gt[gt>0] = 1
    gt = gt.astype(np.bool)

#%% Evaluation seule (sans image affichée)

pred = smartmov.predict(im) # Prédiction avec les images chargées

metric = smartmov.evaluate(pred,gt,metrics_to_compute=['iou','conf','f1'], gt_type='bool') # Evaluation

#%% Evaluation avec image traitée

image_traitee,metric = smartmov.single_display(im,gt=gt,display=['nb_obj','scores','temps'],metrics_to_display=['iou','conf','f1'],
                                                        gt_type='bool',return_scores=True) # Prédiction, évaluation et création de l'image traitée

plt.figure()
plt.imshow(image_traitee) # Affichage de l'image
plt.title('Résultat')

#%% Résultats dans un Excel
RESULTS_DIR = os.path.join(ROOT_DIR,"results/")
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    print("Dossier RESULTS non existant. Création réussie.")

EXCEL_DIR = os.path.join(RESULTS_DIR,"excel/")
if not os.path.exists(EXCEL_DIR):
    os.mkdir(EXCEL_DIR)
    print("Dossier EXCEL non existant. Création réussie.")

EXCEL_FILE = os.path.join(EXCEL_DIR,"res_example.xlsx") # Chemin du Excel à créer ou à modifier

smartmov.create_excel(metric,metrics_to_display=['iou','conf','f1'],nom_fichier=EXCEL_FILE,nom_feuille="Test_evaluate")
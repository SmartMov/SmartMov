import os
import sys
import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from SmartMov import SmartMov

s=(480,720,3) # Shape d'entrée du U-Net

MODELS_DIR = os.path.join(ROOT_DIR,"models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")

TIMESTEP=5

#%% Création du détecteur
smartmov = SmartMov()

#%% Load U-Net
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_skating.h5", shape_unet=s, timestep=TIMESTEP)

#%% Draw U-Net
tf.keras.utils.plot_model(smartmov.unet,show_shapes=True,to_file="unet.png",dpi=320)
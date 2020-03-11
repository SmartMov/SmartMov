# SmartMov

The aim of this network is to be able to detect and identify moving objects in extreme climatic conditions.

This network is the cascading of two different ones.
To perform the motion masks we have created a convolutional LSTM U-Net. For object prediction we use a Mask-RCNN.

The Mask-RCNN is a network capable of adapting to any type of data, which is not the case with our U-Net, which detects movement in the scene. It is therefore necessary to train this network on your dataset so that it is able to adapt. 

***This code has been developed on Tensorflow 2.1.0 and Windows 10, Python=3.7.6***

Dans ce read me, nous vous expliquons comment réaliser l'environnement adéquat et utiliser nos codes. De plus, nous communiquons les différents résultats obtenus sur différentes datasets. Voici la table des matières :

1.  **Création environnement**
    1. Compiler TensorFlow sur GPU
    2. Créer l'environnement  Anaconda
2.  **Fonctionnement des codes**
    1. Entrainement
        + U Net
        + Mask R CNN
    2. Prediction
    3. Evaluation

---
        
# 1. Création environnement
   
## 1.1 Compiler TensorFlow sur GPU


Voici la listes des opérations à suivre utiliser le GPU :
( un fichier que l'on a crée contenant tous les exécutables est disponible au téléchargeant [here](https://drive.google.com/file/d/1bFlTindxlSdjFx2aJfzhhh2NDMO3k3K_/view?usp=sharing) )


*  Vérifier si la carte graphique de l'ordinateur peut supporter (https://developer.nvidia.com/cuda-gpus)
* Mise a jour NVIDIA (GeForce Experience)
*  Installer Cuda 10.1 (executable disponible dans le fichier)
*  Telecharger & Extraire Cudnn pour (zip disponible dans le fichier)
    * Copier les fichiers dans le répertoire : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
* Ajout des variable d'environnement
    * Liste des commandes pour modifier les variables d'environnement à faire dans cmd (admin) :
```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH% 
```

* Executer Executable VisualStudio
	Les 2 VC_redist (disponibles dans le fichier)

## 1.2. Créer l'environnement  Anaconda


Notre codes necessitent  les packages suivants  *imgaug, opencv-python, Pillow, xlwings, numpy, matplotlib, scikit-image, scikit-learn*

Tutoriel d'installation d'environnement viable :
* Environnement
```
conda create IA python=3.7.6
conda activate IA
conda install spyder
```
* Packages
```
pip install imgaug
pip install opencv-python
pip install Pillow
pip install xlwings
conda install numpy
conda install matplotlib
conda install scikit-learn
conda install scikit-image
```
* Tensorflow (pour GPU si l'étape 1.1 a été faite)

```
pip install tensorflow-gpu == 2.1.0
```

Pour vérifier si la compilation GPU se fait bien , écrire dans le prompt de l'environnement créé :

```
python
import tensorflow as tf
tf.test.gpu_device_name()
```
		
Si à la fin il y a écrit  */device:GPU:0*  c'est que cela fonctionne

* Tensorflow (pour CPU si l'étape 1.1 n'a pas été faite)
```
pip install tensorflow == 2.1.0
```
---

# 2. Fonctionnement des codes


Comme nous l'avons décrit précédemment, notre architecture correspond à la mise en cascade de deux réseaux. Un U Net qui permet de détecter le mouvement et un Mask R CNN qui classe les objets. Les fichiers utiles à l'entrainement et l'application de nos réseaux sont disponible dans le dossier **__samples__**. *blabla fonctionnement général*

## 2.1 Entrainement

Les modèles des réseaux entraînés sont disponible dans le fichier **__models__**. Le fichier disponible sur Github contient un lien qui permet de le télécharger (étant trop volumineux pour la plateforme).

### **U Net**

Le **U Net** ne doit être entraîné et ne peut être utilisé que sur la même dataset. Des modèles entraînés pour différentes dataset sont disponibles dans le sous-fichier **__models/Unet__** (skating, PETS2006, pedestrians, blizzard, snowFall, streetCorner, highway, Polytech).
Il vous suffit de les load dans le fichier *blabla comment faire*

Si vous voulez utiliser d'autre dataset il va falloir mettre les images dans le fichier **__dataset__** afin d’entraîner le **U Net**.
Pour l'entrainer il *blabla comment faire*

### **Mask R CNN**

Concernant le **Mask R CNN**, nous l'avons entraîné de façon à qu'il reconnaisse les voitures et les humains. utiliser doit être juste réentrainé si vous voulez rajouter des classes do'bjets à détecter 




---

# 3. Results

We evaluated our network on different sequences of the CD-NET2014 dataset (skating, PETS2006, pedestrians, blizzard, snowFall, streetCornerAtNight, highway) as well as on a video (Polytech) we annotated ourselves.
The metrics we used are IoU and F1-score. The results are presented in the following table:

| Metrics  | skating | PETS2006 | pedestrians | blizzard | snowFall | streetCorner | highway | Polytech | Mean  |
| :------: | :-----: | :------: | :---------: | :------: | :------: | :----------: | :-----: | :------: | :--:  |
| IoU      | 0.816   | 0.82     | 0.62        | 0.75     | 0.56     | 0.38         | 0.64    | 0.62     | 0.651 |
| F1-score | 0.847   | 0.93     | 0.72        | 0.839    | 0.884    | 0.429        | 0.789   | 0.834    | 0.784 |

The visual results are presented in the following video: https://www.youtube.com/watch?v=WBlZlWDwU8s





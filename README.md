# SmartMov'

L’objectif de notre algorithme est de détecter et d’identifier des objets en mouvement dans des mauvaises conditions climatiques.

Notre réseau correspond à la mise en cascade de deux réseaux différents. Nous avons créé un U Net LSTM convolutif  afin de créer les masques de mouvement. Pour classer les objets nous utilisons un Mask-RCNN.

Le Mask-RCNN est un réseau capable de s'adapter à tout type de données, ce qui n'est pas le cas de notre U-Net, qui détecte les mouvements sur la scène. Il est donc nécessaire de former ce réseau sur votre jeu de données afin qu'il soit capable de s'adapter.

***Le code à été développé pour code Tensorflow 2.1.0, Windows 10 et Python=3.7.6***

Dans ce read me, nous vous expliquons comment réaliser l'environnement adéquat et utiliser nos codes. De plus, nous communiquons les différents résultats obtenus sur différentes datasets. Voici la table des matières :

1. **Création environnement** 
    1. Compiler TensorFlow sur GPU
    2. Créer l’environnement  Anaconda
2. **Fonctionnement des codes**
    1. Entrainement
        + Mask-RCNN
        + U Net
    2. Prédiction
    3. Evaluation
3. **Resultats**

---
<br>      
  
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
Les 2 VC_redist (disponibles dans le fichier``)

*Un tuto de l’installation précédente est donné sur cette page https://www.tensorflow.org/install/gpu, nous l’avons simplifié et amélioré dans ce readme*

## 1.2. Créer l'environnement  Anaconda


Notre programme nécessite les packages suivants  *imgaug, opencv-python, Pillow, xlwings, numpy, matplotlib, scikit-image, scikit-learn*

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
<br>

# 2. Fonctionnement des codes


Comme nous l'avons décrit précédemment, notre architecture correspond à la mise en cascade de deux réseaux. Un U Net qui permet de détecter le mouvement et un Mask-RCNN qui classe les objets. Les fichiers utiles à l'entrainement et l'application de nos réseaux sont disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples), voici leur utilité :
* [*draw_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/draw_unet.py) : dessine l’architecture du **U Net**
* [*train_rcnn.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_rcnn.py) : entraîne le model du **Mask-RCNN** (à utiliser pour rajouter des types d’objets à détecter)
* [*train_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_unet.py) : entraîne le model du **U Net** (à utiliser pour appliquer l’algorithme sur une nouvelle dataset)
* [*predict.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/predict.py) : applique l’algorithme à une dataset, need les modèles des deux réseaux entraînés
* [*evaluate.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/evaluate.py) : applique l’algorithme à une dataset contenant des groundtruth afin de l’évaluer, need les modèles 

<br>

## 2.1 Entraînement

Les modèles des réseaux entraînés sont disponibles dans le fichier [**models**](https://github.com/SmartMov/SmartMov/tree/master/models). Le fichier disponible sur Github contient un lien qui permet de le télécharger (étant trop volumineux pour la plateforme).

<br>

### **Mask-RCNN**

Concernant le **Mask R CNN**, nous l'avons entraîné de façon à qu'il reconnaisse les voitures et les humains. Son entrainement est présent dans le fichier [*train_rcnn.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_rcnn.py) disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples),   il est basé sur les idées de l'article suivant : https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d. 

Il doit être réentrainé si vous voulez rajouter des classes d’objets à détecter. Cela se fait de la manière suivante :
* Dans la classe *InferenceConfig* il faut ajouter   ...
* *blabla*

<br>

### **U Net**

Le **U Net** ne doit être entraîné et ne peut être utilisé que sur la même dataset. Des modèles entraînés pour différentes dataset sont disponibles dans le sous-fichier **models/U-Net** (skating, PETS2006, pedestrians, blizzard, snowFall, streetCorner, highway, Polytech).

Si vous voulez **utiliser d'autre dataset** il va falloir mettre les images dans le fichier [**dataset_train**](https://github.com/SmartMov/SmartMov/tree/master/dataset_train) afin d’entraîner le **U Net**.
Pour l’entraîner sur une nouvelle dataset cela se passe dans le fichier [*train_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_unet.py) *blabla comment faire*  *se faire de deux façons : en utilisant un objet DataGenerator ou en spécifiant simplement un dossier contenant les images et les vérités de base à utiliser pour la formation*
*vérifier seuil*
Néanmoins, si vous voulez **améliorer un model** déjà entraîné, cela est possible,vous pouvez le load et utiliser la fonction ...

<br>

## 2.2 Prédiction

La prédiction se fait dans le fichier [*predict.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/predict.py) disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples).
Pour effectuer la prédiction il faut placer les images à tester dans le fichier [**dataset_test/input**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/input).

S'il y a eu un ajout de classes pour le Mask-RCNN il faut la rajouter à la suite dans le tableau du fichier ```class_names = ['BG','person','car']``` ainsi que rajouter le nombre de classes supplémentaires dans ```NUM_CLASSES=1+2```

Il faut modifier dans la fonction : ```smartmov.load_models ( ... )``` le nom du model du **U Net** afin de le faire correspondre à la dataset à traiter  (et aussi modifier celui du **Mask-RCNN** si vous l’avez ré entrainé).

La prédiction d’une image correspond à la superposition de la même image et des différents masques de couleur ainsi que la boîte englobante des objets prédis en mouvement. Nous ajoutons à cela en haut de la box le nom de la classe et sa probabilité prédite d’appartenance. Nous instaurez un notion de tracking pour que  chaque objet garde la même couleur tout au long de la scène.


<br>

## 2.3 Evaluation

L'évaluation se fait de la même façon que la prédiction à la différence qu'elle nécessite les vérités terrains, cela s'effectue dans le fichier [*evaluate.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/evaluate.py). Il faut placer les images brutes à tester dans le dossier [**dataset_test/input**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/input) et les groundtruth dans [**dataset_test/groundtruth**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/groundtruth). Le programme va donc comparer la prédiction avec les vérités (ne pas oublier de compiler les bons models). Pour jauger le réseau nous utilisons différentes métriques :
* IoU  et F1 Score 
* La matrice de confusion (matrice 2 x 2)

| Number of pixels **correctly detected** as « movement » | Nb of pixels which **should have been detected** as « movement » |
| :------: | :-----: | 
| Nb of pixels which **shouldn’t have been detected** as « mouvement » | Nb of pixels **correctly detected** as  « non movement » |

<br>
Ce fichier donne donc :

* Les prédictions de toutes les images mis dans le dossier (section *Evaluation* du fichier)

* Une nouvelle feuille d'un tableau excel contenant les résultats des différentes métriques ainsi que leurs évolutions temporelles (section *Excel création*)

* Une vidéo dont on peut régler le nombre de fps qui correspond à la concaténation de toutes les images prédites. Il faut que chaque frame de la vidéo est annotée; en haut à gauche : le nombre le temps d'inférence, le numéro de l'image et le temps d"inférence & en bas à droite : le score IoU et F1 pour l'image en question (section *Video*)

---

<br>

# 3. Resultats

Nous avons évalué notre réseau sur différentes séquences de l'ensemble de données CD-NET2014 (skating, PETS2006, pedestrians, blizzard, snowfall, streetCornerAtNight, highway) ainsi que sur une vidéo (Polytech) que nous avons nous-mêmes annotée.
Les mesures que nous avons utilisées sont l'IoU et le F1. Les résultats sont présentés dans le tableau suivant :

| Métriques  | skating | PETS2006 | pedestrians | blizzard | snowFall | streetCorner | highway | Polytech | Mean  |
| :------: | :-----: | :------: | :---------: | :------: | :------: | :----------: | :-----: | :------: | :--:  |
| IoU      | 0.816   | 0.82     | 0.62        | 0.75     | 0.56     | 0.38         | 0.64    | 0.62     | 0.651 |
| F1-score | 0.847   | 0.93     | 0.72        | 0.839    | 0.884    | 0.429        | 0.789   | 0.834    | 0.784 |

Les résultats visuels sont disponibles dans la vidéo YouTube suivante : https://www.youtube.com/watch?v=WBlZlWDwU8s

---

<br>

> Pour toute question supplémentaire n’hésitez pas à nous contacter. 
>
> Nos contacts :
> 
> Mail : serranomael@gmail.com & jacques1434@gmail.com
> 
> LinkedIn : www.linkedin.com/in/jacques-budillon & www.linkedin.com/in/mael-serrano

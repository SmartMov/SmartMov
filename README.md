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


Voici la listes des opérations à suivre pour utiliser le GPU :
( un dossier "Install - Env" que l'on a créé contenant tous les exécutables est disponible au téléchargement [ici](https://drive.google.com/file/d/1pZp3c-4b2EdLtvZ1tkSsjoLVNAqNapv_/view?usp=sharing) )


*  Vérifier si la carte graphique de l'ordinateur peut supporter (https://developer.nvidia.com/cuda-gpus)
*  Mise a jour NVIDIA (GeForce Experience)
*  Installer Cuda 10.1 (executable disponible dans le dossier "Install - Env")
*  Telecharger & Extraire Cudnn (zip disponible dans le dossier "Install - Env")
    * Copier les fichiers dans le répertoire : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
* Ajout des variable d'environnement
    * Liste des commandes pour modifier les variables d'environnement, à faire dans cmd (admin) (sur Windows) :

```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

* Executer Executable VisualStudio
Les 2 VC_redist (disponibles dans le dossier "Install - Env")

* Pycocotools
    * Installer Visual C++ 2015 Build Tools (fichier [vstudio2015.exe](https://go.microsoft.com/fwlink/?LinkId=691126), exécutable disponible dans le dossier "Install - Env")
    * Aller à *C:\Program Files (x86)\Microsoft Visual C++ Build Tools* et lancer *vcbuildtools_msbuild.bat*
    * Lancer la commande (possibles messages d'erreur concernant des conflits de version, pas important ici) :
    ```
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    ```
    * Si la commande se termine par les lignes ci-dessous tout est bon
    ```
    Installing collected packages: pycocotools
    Successfully installed pycocotools-2.0
    ```
    Si un problème est survenu lors de cette étape, voir ce [*lien*](https://github.com/philferriere/cocoapi).

*Un tutoriel de cette installation est donné sur cette page https://www.tensorflow.org/install/gpu, nous l’avons simplifié et amélioré dans ce readme*

## 1.2. Créer l'environnement Anaconda


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

Pour vérifier si la compilation GPU se fait bien, écrire dans le prompt de l'environnement créé :

```
python
import tensorflow as tf
tf.test.gpu_device_name()
```

Si les résultat est */device:GPU:0*, cela signifie que tout fonctionne

* Tensorflow (pour CPU si l'étape 1.1 n'a pas été faite)
```
pip install tensorflow == 2.1.0
```

---
<br>

# 2. Fonctionnement des codes


Comme nous l'avons décrit précédemment, notre architecture correspond à la mise en cascade de deux réseaux. Un U-Net qui permet de détecter le mouvement, ainsi qu'un Mask-RCNN qui segmente et classe les objets. De plus, nous avons rajouté un suvi des formes afin que la couleur du masque d'un objet soit la même durant plusieurs images consécutives. Les fichiers utiles à l'entrainement et l'application de nos réseaux sont disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples), voici leur utilité :
* [*create_video.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/create_video.py) : applique l'algorithme à plusieurs images afin de réaliser l'évaluation et de créer la vidéo des résultats en sortie
* [*draw_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/draw_unet.py) : dessine l’architecture du **U-Net** (nécessite Graphviz)
* [*train_rcnn.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_rcnn.py) : entraîne le model du **Mask-RCNN** (à utiliser pour rajouter des types d’objets à détecter)
* [*train_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_unet.py) : entraîne le model du **U-Net** (à utiliser pour appliquer l’algorithme sur une nouvelle dataset)
* [*predict.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/predict.py) : applique l’algorithme à une dataset, il faut auparavant que les deux réseaux soient entraînés
* [*evaluate.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/evaluate.py) : applique l’algorithme à une dataset contenant des groundtruth afin de l’évaluer, il faut également que les deux réseaux soient entraînés

**Nous allons à présent détailler comment utiliser nos fichiers de programmation. Il faut tout de même noter que nous avons effectué un docstring de TOUTES les fonctions (lors de leur appel une documentation des paramètres entrées/sorties est disponible), ceci permet d'alléger ce README.** 

<br>

## 2.1 Entraînement

Les modèles des réseaux entraînés sont disponibles dans le fichier [**models**](https://github.com/SmartMov/SmartMov/tree/master/models). Ce fichier contient un lien qui permet de télécharger tous les modèles (étant trop volumineux pour être uploadés ici).

<br>

### **Mask-RCNN**

Concernant le **Mask-RCNN**, nous l'avons entraîné de façon à ce qu'il reconnaisse les voitures et les humains. La méthode d'entrainement que nous avons utilisée est détaillée dans le fichier [*train_rcnn.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_rcnn.py) disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples), elle est basée sur les idées de l'article suivant : https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d. 

Il doit être ré-entrainé afin de rajouter des classes d’objets à détecter. Cela se fait de la manière suivante :
* Dans la classe *InferenceConfig* il faut modifier l'attribut NUM_CLASSES qui devra valoir le nombre de classes que vous souhaitez détecter + 1 (pour le background).
* La variable *class_names* doit également être modifiée et sa longueur doit être la même que NUM_CLASSES. *class_names* correspond aux noms des classes à détecter. Le premier élément doit être 'BG'.
* Une classe doit être créé sur le modèle de la classe utils.Dataset. Le détail de la création de cette classe est ci-dessous.
* L'objet correspondant à cette classe doit être créé puis la dataset doit être chargée en utilisant la méthode *load_dataset()* crée ci-dessus. La dataset doit ensuite être préparée avec la méthode *prepare()*
* Il faut ensuite créer le modèle RCNN avec *smartmov.create_model()*. Les paramètres à utiliser sont 'model_dir' qui correspond à la localisation des logs, 'config' qui doit contenir l'objet créé avec la classe InferenceConfig, 'reuse'=True pour réutiliser un ancien modèle, 'model_rcnn' doit contenir le fichier .h5 contenant les poids du modèle RCNN déjà entrainé et 'class_names' doit être la liste des noms de classes crée plus haut.
* Ensuite la méthode *smartmov.train()* doit être utilisée afin d'entrainer réellement le Mask-RCNN. Les paramètres sont 'dataset_train_rcnn' qui correspond à l'objet créé au-dessus contenant les images pour le training, 'dataset_val_rcnn' est le même type d'objet mais contenant les images pour la validation, 'epochs_rcnn' correspond au nombre d'epochs pour l'entrainement, et 'layers'='heads' pour ne pas réentrainer tous les poids du réseau (inutile puisqu'un modèle pré-entrainé a été chargé)
* Une fois l'entrainement terminé, il est important de convertir le modèle en mode 'inference' avant de pouvoir faire des prédictions. Pour ce faire, il faut utiliser la méthode *smartmov.convert_rcnn()*
* Il est enfin possible d'enregistrer le modèle en utilisant *smartmov.save()* avec pour paramètres 'models_to_use'='rcnn' et 'dir_rcnn' le chemin du fichier .h5 ou enregsitrer les poids.

#### Création de la classe pour l'entrainement du Mask-RCNN
Pour entrainer le Mask-RCNN, il faut créer une classe qui permettra de renvoyer tous les masques et les images qui leur correspondent. Pour ce faire, deux méthodes doivent être déclarées. La première est *load_dataset()*. Elle doit fonctionner comme ceci :
* Pour chaque classe que l'on souhaite détecter, il faut appeller la méthode *add_class()* de *utils_Dataset* afin que l'objet créé ensuite connaisse toutes les classes qu'il devra détecter
* Pour chaque image de la dataset, il faut appeller la méthode *add_image()*. De cette manière, lorsque nous appelerons la méthode *load_image()* à partir de l'indice de l'image dans la dataset, toutes les informations de l'image seront connues.

La seconde méthode à déclarer est *load_mask()*. Cette méthode devra prendre en paramètre l'indice de l'image dans la dataset et renvoyer le masque correspondant. La sortie de cette fonction doit être un tuple de taille 2, comprenant un tableau de taille (H,W,nb_instances) et un autre tableau correspondant à l'indice de la classe de chaque instance.

La classe a par exemple été crée pour la Dataset Davis, et le détail du code est présent dans le fichier [*davis.py*](https://github.com/SmartMov/SmartMov/blob/master/davis.py).

Un autre exemple de création de la classe est à trouver à cette [*URL*](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).


<br>

### **U-Net**

Le **U-Net** ne peut être utilisé que sur des images semblables à celles de la dataset sur laquelle il s'est entrainé. Des modèles entraînés pour différentes dataset sont disponibles dans le sous-dossier **models/U-Net** (skating, PETS2006, pedestrians, blizzard, snowFall, streetCorner, highway, Polytech).

Si vous voulez **utiliser d'autres datasets** il faut mettre les images dans le fichier [**dataset_train**](https://github.com/SmartMov/SmartMov/tree/master/dataset_train) afin d’entraîner le **U-Net**.
Pour l’entraîner sur une nouvelle dataset, les détails sont dans le fichier [*train_unet.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/train_unet.py). Les différentes étapes présentées dans ce code sont les suivantes :
* Création du modèle avec *smartmov.create_model()*. Les paramètres sont 'unet' pour spécifier qu'il s'agit de ce type de modèle qui est à créer, 'shape_unet' doit être un tuple donnant la taille des images en entrée du réseau, et 'timestep' est le nombre d'images consécutives à utiliser pour estimer le mouvement.
* Entrainement du modèle : Il existe deux méthodes d'entrainement :
    * La première consiste à utiliser un objet de type *DataGenerator* qui dans un premier temps charge les images avec *DataGenerator.load_data()* (qui prend en paramètre une liste de dossiers contenant les images et leurs groundtruth dans des dossiers "input" et "groudntruth" au sein de chacun des élements de la liste ainsi que le nombre d'images à utiliser parmi tous ces dossiers). Dans ce cas, lorsque la méthode *smartmov.train()* est appellée, le paramètre 'generator_unet' doit contenir cet objet créé avant. Les autres paramètres sont les classiques epochs et batch_size.
    * La seconde consiste à simplement passer en paramètre de la méthode *smartmov.train()* le paramètre 'dir_train_unet' (qui correspond à un dossier organisé de la même manière que pour la première méthode) ainsi que le batch_size et le nombre d'epochs.
* Une fois l'entrainement terminé, le modèle peut être sauvegardé en utilisant la méthode *smartmov.save()* avec les paramètres 'models_to_save'='unet' et 'dir_unet' correspond au fichier .h5 ou le modèle sera sauvegardé.

Néanmoins, il est également possible d'**améliorer un modèle** déjà entraîné, il faut pour ce faire d'abord utiliser la méthode *smartmov.load_model()* en chargeant le modèle à améliorer puis ensuite utiliser la méthode *smartmov.train()* de la même manière que décrit ci-dessus (sans utiliser *smartmov.create_model()*).

<br>

## 2.2 Prédiction

La prédiction se fait dans le fichier [*predict.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/predict.py) disponible dans le dossier [**samples**](https://github.com/SmartMov/SmartMov/tree/master/samples).
Pour effectuer la prédiction il faut placer les images à tester dans le fichier [**dataset_test/input**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/input).

L'objet *InferenceConfig* créé au début du fichier doit être modifié pour être adapté au Mask-RCNN. Il faut que l'attribut ```NUM_CLASSES``` vale le nombre de classes différentes à détecter +1, et que la variable ```class_names=['BG',...]``` contienne tous les noms des classes à prédire.

Il faut modifier dans la fonction : ```smartmov.load_models ( ... )``` le nom du fichier contenant le modèle du **U-Net** afin de le faire correspondre à la dataset à traiter  (et aussi modifier celui du **Mask-RCNN** si vous l’avez ré-entrainé).

La prédiction d’une image correspond à la superposition de la même image et des différents masques de couleur ainsi que la boîte englobante des objets prédis en mouvement. Nous ajoutons à cela en haut de la box le nom de la classe et sa probabilité prédite d’appartenance. Nous instaurez un notion de tracking pour que  chaque objet garde la même couleur tout au long de la scène.


<br>

## 2.3 Evaluation

L'évaluation se fait de la même façon que la prédiction à la différence qu'elle nécessite les vérités terrains, cela s'effectue dans le fichier [*evaluate.py*](https://github.com/SmartMov/SmartMov/blob/master/samples/evaluate.py). Il faut placer les images brutes à tester dans le dossier [**dataset_test/input**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/input) et les groundtruth dans [**dataset_test/groundtruth**](https://github.com/SmartMov/SmartMov/tree/master/dataset_test/groundtruth). Le programme va donc comparer la prédiction avec les vérités (ne pas oublier de compiler les bons modèles comme pour la prédiction). Pour jauger le réseau nous utilisons différentes métriques :
* Si la vérité terrain est un simple masque binaire :
    * IoU  et F1 Score
    * La matrice de confusion (matrice 2 x 2)
* Si la vérité terrain est un découpage de chaque instance :
    * IoU et F1 Score
    * Matrice de confusion (2 x 2)
    * Correspondances de chacun des rectangles (IoU sur les paires de rectangles)
    * Correspondances de chacun des masques (IoU sur les paires de masques)
* Nous avons créé une métrique particulière qui permet d'évaluer la classification des objets pour les datasets que nous avons utilisées (c'est une "F1 score" des prédictions pondérés par leurs probalités) *elle est appellée class*

| Nombre de pixels **correctement détectés** comme « en mouvement » | Nombre de pixels qui **auraient du être détectés** comme « en mouvement » |
| :------: | :-----: | 
| Nombre de pixels qui **n'auraient pas du être détectés** comme « en mouvement » | Nombre de pixels **correctement détectés** comme « sans mouvement » |

<br>
Ce fichier donne donc :

* Les prédictions de toutes les images mises dans le dossier (section *Evaluation* du fichier)

* Une nouvelle feuille d'un tableau Excel contenant les résultats des différentes métriques ainsi que leurs évolutions en fonction des images de la séquence (section *Excel création*)

* Une vidéo dont on peut régler le nombre de fps qui correspond à la concaténation de toutes les images prédites. Chaque frame de la vidéo est annotée, avec en haut à gauche : le nombre d'objets détectés, le temps d'inférence et le numéro de l'image dans la séquence & en bas à droite : le score IoU et F1 pour l'image en question (section *Video*)

Voici des exemples de l'application de cet algorithme sur des images :
![alt text](https://github.com/SmartMov/SmartMov/blob/master/test_images/Res_photos_git.png)

---

<br>

# 3. Résultats

Nous avons évalué notre réseau sur différentes séquences de l'ensemble de données CD-NET2014 (skating, PETS2006, pedestrians, blizzard, snowfall, streetCornerAtNight, highway) ainsi que sur une vidéo (Polytech) que nous avons nous-mêmes annotée.
Les métriques que nous avons utilisées sont l'IoU et le F1 ainsi que celle que nous avons créé évaluant la classification des objets. Les résultats sont présentés dans le tableau suivant :

| Métriques  | skating | PETS2006 | pedestrians | blizzard | snowFall | streetCorner | highway | Polytech | Mean  |
| :------: | :-----: | :------: | :---------: | :------: | :------: | :----------: | :-----: | :------: | :--:  |
| IoU      | 0.75   | 0.82     | 0.68        | 0.78     | 0.89     | 0.44         | 0.71    | 0.62     | 0.73 |
| F1-score | 0.87   | 0.85     | 0.76        | 0.87    | 0.94    | 0.59        | 0.83   | 0.84    | 0.82 |
| Classification-score | 0.93   | 0.97     | 0.96        | 0.98    | 0.59    | 0.59        | 0.91   | 0.85    | 0.89 |

Deux tableurs Excel contenant ces resultats sont disponibles :
* Un tableur détaillé [(lien)](https://drive.google.com/open?id=1Mx1Sb7iwX_EZndNRi91E9cjNKQWcqCLx) qui a pour chaque dataset une sheet dédiée regroupant :
    * La matrice de confusion + recal & accuracy
    * Les résultats des differentes métriques avec leur évolution au cours des images 
    * Les différentes intervalles de confiance de chaque scores
* Un tableur simplifié [(lien)](https://drive.google.com/file/d/11GBzR1UchXBRhhNger14mBBRmljKEKbc/view?usp=sharing) qui regroupe toutes les datasets

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

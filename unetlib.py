import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.layers import TimeDistributed
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import glob
from PIL import Image

def create_template_in(nom_fichier,f):
  temp_in = nom_fichier
  template = temp_in+"/"+f+"/input/*"
 
  return sorted(glob.glob(template))

def create_template_gt(nom_fichier,f):
  temp_in = nom_fichier
  template = temp_in+"/"+f+"/groundtruth/*"
  
  return sorted(glob.glob(template))

def gen_batch(batch_size,timestep,nom_fichier,s,f='Train'):
    '''
    Parameters
    ----------
    batch_size : Taille du lot
    timestep : Nombre d'images dans la séquence
    nom_fichier : Nom du fichier des images
    s : Shape des images
    f : Validation ou Train The default is 'Train'.


    Yields
    ------
    batch_inputs : Batch images en entrée
    batch_targets : Batch images masque

    '''

    T_in=create_template_in(nom_fichier,f)
    T_gt=create_template_gt(nom_fichier,f)
   
    nb_im = len(T_gt)
    i=0
    while i < (nb_im-timestep)//batch_size:
        batch_inputs=np.zeros((batch_size,timestep,s[0],s[1],s[2]),dtype=np.float32)
        batch_targets=np.zeros((batch_size,s[0],s[1],1),dtype=np.uint8)
        for j in range(batch_size):
            for k in range(timestep):
                inp = Image.open(T_in[i*batch_size+j+k])
                inp = inp.resize((s[1],s[0]))
                batch_inputs[j,k] = ((np.array(inp)-127.5)/127.5).astype(np.float32)
            gt = Image.open(T_gt[i*batch_size+j+k])
            gt = np.array(gt.resize((s[1],s[0])),dtype=np.uint8)
            gt = np.expand_dims(gt,axis=-1)
            if gt.dtype!=np.bool:
                gt[(gt<255) & (gt>1)] = 0
                gt[gt>0] = 1
                gt = gt.astype(np.bool)
            batch_targets[j]=gt

        i+=1
        if i>=(nb_im-timestep)//batch_size:
            i=0
    
        yield batch_inputs,batch_targets
    
    
def create(s,TIMESTEP):
    
    
    '''
    Parameters
    ----------
    s : Shape des images.
    TIMESTEP : Nombre d'images pour la séquence (récurrence)
    
    Returns
    -------
    model : Le model du réseau compilé

    '''    
        
    input_img  =  tf.keras.layers.Input ( shape = (TIMESTEP,s[0],s[1],s[2]), name='input_img')
    
    x1 = tf.keras.layers.ConvLSTM2D(filters=6,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=True,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv1') ( input_img )
    
    x1_bis = tf.keras.layers.ConvLSTM2D(filters=6,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=False,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv1_pont') ( input_img )
    
    x = TimeDistributed(tf.keras.layers.BatchNormalization(momentum=0.8), name='batchnorm_conv1')(x1)
    x = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2), name='pooling_conv1') (x)
    
    x2 = tf.keras.layers.ConvLSTM2D(filters=12,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=True,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv2')(x)
    
    x2_bis = tf.keras.layers.ConvLSTM2D(filters=12,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=False,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv2_pont')(x)
    
    x = TimeDistributed(tf.keras.layers.BatchNormalization(momentum=0.8), name='batchnorm_conv2')(x2)
    x = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2), name='pooling_conv2')(x)
    
    x3 = tf.keras.layers.ConvLSTM2D(filters=18,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=True,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv3') (x)
    
    x3_bis = tf.keras.layers.ConvLSTM2D(filters=18,activation='tanh',
                                         kernel_size=3,padding='same',
                                         return_sequences=False,
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv3_pont') (x)
    
    x = TimeDistributed(tf.keras.layers.BatchNormalization(momentum=0.8), name='batchnorm_conv3')(x3)
    x = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2), name='pooling_conv3')(x)
    
    x = tf.keras.layers.ConvLSTM2D(filters=18,activation='tanh',
                                         kernel_size=3,padding='same',
                                         recurrent_activation='sigmoid',
                                         recurrent_dropout=0,
                                         unroll=False,use_bias=True,
                                         name='conv4') (x)
    
    x = tf.keras.layers.Conv2DTranspose(12, 3, padding='same', strides=2,activation='relu', name='convT1') (x)
    x4 = tf.keras.layers.BatchNormalization(momentum=0.8, name='batchnorm_convT1') (x)  
    
    pont1 = tf.concat([x3_bis,x4],axis=-1)
    
    x5 = tf.keras.layers.Conv2DTranspose(12, 3, padding='same', strides=2,activation='relu', name='convT2') (pont1)
    x5 = tf.keras.layers.BatchNormalization(momentum=0.8, name='batchnorm_convT2') (x5)
    
    pont2 = tf.concat([x2_bis,x5],axis=-1)
    
    x6 = tf.keras.layers.LeakyReLU() (pont2)
    x = tf.keras.layers.Conv2DTranspose(6, 3, strides=2, padding='same', activation='relu', name='convT3') (x6)
    
    x7 = tf.keras.layers.BatchNormalization(momentum=0.8, name='batchnorm_convT3') (x)
    
    pont3 = tf.concat([x1_bis,x7],axis=-1)
    
    x8 = tf.keras.layers.LeakyReLU() (pont3)
    y = tf.keras.layers.Conv2DTranspose(2, 3, padding='same', activation='softmax', name='output') (x8)
    
    model = tf.keras.models.Model(inputs=input_img,outputs=y, name='unet')
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    model.compile(optimizer=optimizer,loss=loss_object,metrics=[metric],experimental_run_tf_function=False)
    
    return model

def train (BATCH_SIZE,TIMESTEP,EPOCHS,s,model,checkpoint_dir,nom_fichier=None,generator=None):
    '''
    
    Parameters
    ----------
    BATCH_SIZE : Taille du lot 
    TIMESTEP : Nombre d'images dans la séquence
    EPOCHS : Nombre d'époques
    nom_fichier : Nom du fichier des images
    generator : Generateur contenant les images pour l'entrainement
    checkpoint_dir : string qui représente l'emplacement des checkpoints lors de l'entrainement
    s : Shape des images
    model : Modèle du réseau créé

    Returns
    -------
    model : Modèle du réseau entrainé

    '''
    assert (nom_fichier!=None or generator!=None), "Pas de données pour entrainer"
    
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir)
    
    if generator!=None:
        gen = generator
        model.fit(gen.generate_train(BATCH_SIZE),steps_per_epoch=gen.steps_per_epoch,
                  epochs=EPOCHS,validation_data = gen.generate_val(BATCH_SIZE),
                  validation_steps = gen.get_nb_val()//BATCH_SIZE,
                  callbacks=[ckpt])
    else:
        temp_in = nom_fichier
        template = temp_in+"/Train/input/*"
        nb_images_training=len(glob.glob(template))
        
        temp_in = nom_fichier
        template = temp_in+"/Validation/input/*"
        nb_images_valid=len(glob.glob(template))
        
        model.fit(gen_batch(BATCH_SIZE,TIMESTEP,nom_fichier,s,'Train'),steps_per_epoch=(nb_images_training-TIMESTEP)//(BATCH_SIZE),
                  epochs=EPOCHS,validation_data = gen_batch(BATCH_SIZE,TIMESTEP,nom_fichier,s,'Validation'),
                  validation_steps = (nb_images_valid-TIMESTEP)//(BATCH_SIZE),
                  callbacks=[ckpt])
    
    return model
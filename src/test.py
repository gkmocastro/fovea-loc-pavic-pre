import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from keras.models import load_model
from dataset import ler_img_dataset
from utils import iou, dice_coef, dice_coef_loss
import numpy as np
import cv2
#
# gpu_device = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu_device, False)

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}):
  modelo_teste = tf.keras.models.load_model('models\\unet_dice.keras')
  #modelo_teste = load_model('models\\unet_bce.h5')

  #modelo_teste.load_weights('/content/modelo_drive_40.h5')


#predict

def segmenta_img(img, modelo_teste):
  predicao = modelo_teste.predict(np.expand_dims(img, axis=0))[0]
  predicao = predicao > 0.5 #estudo do limiar between 
  predicao = predicao.astype(np.int32)
  predicao = np.squeeze(predicao, axis=-1)
  return predicao

#for de 0.0 ate 1.0 para limiar



'''
tf.keras.utils.load_img(
    path,
    grayscale=False,
    color_mode='rgb',
    target_size=None,
    interpolation='nearest',
    keep_aspect_ratio=False
)
'''

#image = ler_img_dataset('data\\dataset_final\\train\\images\\aria_c_resized_2.png')
img = cv2.imread(r'C:\Users\gkhay\Workstation\mestrado\Pesquisa\macula_seg\data\dataset_final\train\images\aria_c_img_019.png')
maskgt = cv2.imread(r'C:\Users\gkhay\Workstation\mestrado\Pesquisa\macula_seg\data\dataset_final\train\masks\aria_c_img_019.png')
img = cv2.resize(img, (256,256))
img = img / 255.0
img = img.astype(np.float32)
#input_arr = tf.keras.utils.img_to_array(image)
#input_arr = np.array([input_arr])  # Convert single image to a batch.
#predictions = modelo_teste.predict(input_arr)
pred = segmenta_img(img, modelo_teste)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(maskgt, cmap='gray')
plt.subplot(133)
plt.imshow(pred,cmap='gray')
plt.show()
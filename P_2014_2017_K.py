import csv
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import os
# from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input,Reshape,Add
from keras.layers import Dense, Flatten, Dropout,DepthwiseConv2D, Layer, GlobalAveragePooling1D, Input, RandomCrop, RandomFlip, Embedding,LayerNormalization,Activation
# from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers.normalization.batch_normalization_v1 import BatchNormalization
# from tensorflow.keras.layers import normalization
from keras.layers import BatchNormalization
# from tensorflow.keras.utils.data_utils import get_file
from keras.utils import get_file
from keras import backend as K
from scipy.io import loadmat
import seaborn as sn
import pandas as pd
# import tensorflow as tf
from keras import layers, models
from keras.losses import categorical_crossentropy
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,KFold
import matplotlib.pyplot as plt
from keras.utils import to_categorical, plot_model
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import numpy as np
import os
from keras.utils import to_categorical # np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
# from keras.activations import softmax
EPOCHS = 60 # 100  # ---
EPOCHS_LENET = 10
IMG_WIDTH = 56 # 128
IMG_HEIGHT = 56 # 128
NUM_CATEGORIES = 11
INPUT_IMAGE_DIR = "images/VI_images_test/2017Sub_one/" 
INPUT_MODEL_DIR = "models"
OUTPUT_MODEL_DIR = "models"
SAVE_DIR = "images"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#-------- dimensions ---------
img_rows, img_cols = 56, 56
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
#-----------------------------

train_size = 50000
#--- coarse 1 classes ---
num_c_1 = 2
#--- coarse 2 classes ---
num_c_2 = 4
#--- fine classes ---
num_classes = 11

batch_size = 128
epochs = 80  # 75 or 90

#--- file paths ---
log_filepath = './tb_log_medium_dynamic/'
weights_store_filepath = './models/'  
train_id = '1'
model_name = 'BCNN_SVHN'+train_id
model_path = os.path.join(weights_store_filepath, model_name)

def scheduler(epoch):
  learning_rate_init = 0.003
  if epoch > 50:
    learning_rate_init = 0.001  # 0.0015   
  if epoch > 70:
    learning_rate_init = 0.0008  # 0.0001  
  return learning_rate_init

class LossWeightsModifier(tf.keras.callbacks.Callback):
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 8:
      K.set_value(self.alpha, 0.2)
      K.set_value(self.beta, 0.7)
      K.set_value(self.gamma, 0.1)
    if epoch == 18:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.2)
      K.set_value(self.gamma, 0.7)
    if epoch == 38:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.1)
      K.set_value(self.gamma, 0.8)
    if epoch == 58:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 0)
      K.set_value(self.gamma, 1)
def import_model():
    print('please import the well-trained model：')
    filename = input("\nModel name: ")
    model = tf.keras.models.load_model(f"{INPUT_MODEL_DIR}/{filename}")  # {filename}.h5")
    model.summary()
    return model
def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category.\
    Inside each category directory will be some number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 1. `labels` should
    be a list of strings (ex: 'Laptop', 'Blender', ...), \
    representing the categories for each of the
    corresponding `images/loads`.
    """
    images = []
    labels = []
    labels_literal = []
    for category in os.listdir(data_dir):  
        for img_file in os.listdir(os.path.join(data_dir, category)):  
            img = cv2.imread(os.path.join(data_dir, category, img_file)) 
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = np.array(img) 
            images.append(img)
            labels.append(str(category))  
        labels_literal.append(str(category)) 
    return (images, labels), labels_literal
def load_data_1(data_dir):
    Class = ['Air Conditioner', 'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hairdryer', 'Heater',
             'Incandescent Light Bulb', 'Laptop', 'Microwave', 'Vacuum','Washing Machine']
    print('Class:', Class)
    IMG_SZ = 56
    trn_data = []
    def create_training_data():
        for cat in Class:
            path = os.path.join(data_dir, cat)
            class_num = Class.index(cat)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))
                    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)  
                    new_arr = cv2.resize(img_arr, (IMG_SZ, IMG_SZ))
                    # print('new_arr:',new_arr.shape)
                    trn_data.append([new_arr, class_num])
                except Exception as e:
                    pass
    create_training_data()
    setlen = len(trn_data)
    print(setlen)
    X = []
    y = []
    for image, label in trn_data:
        X.append(image)
        y.append(label)
    return (X, y), Class
def process_data_VI_Images(k_folds=True):
    le = preprocessing.LabelEncoder()
    (images, labels), labels_literal = load_data_1(INPUT_IMAGE_DIR) 
    X = np.array(images)
    print('X[0]-----before:', X[0])
    X = X/255 
    print('X[0]-----after:', X[0])
    Y = np.array(labels)
    skf = StratifiedKFold(n_splits=10, shuffle=True)  # -------------------------------------
    # k = 10
    # skf = KFold(k)
    i = 0
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_train_literal = []
    Y_test_literal = []
    for train_index, test_index in skf.split(X, Y):  
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        Y_train_literal.append(Y[train_index])
        Y_test_literal.append(Y[test_index])

        X_train[i] = X_train[i].reshape(X_train[i].shape[0], IMG_HEIGHT, IMG_WIDTH, 3) 
        X_test[i] = X_test[i].reshape(X_test[i].shape[0], IMG_HEIGHT, IMG_WIDTH, 3)  
        le.fit(Y_test_literal[i])  
        Y_test.append(le.transform(Y_test_literal[i]))  
        le.fit(Y_train_literal[i])  
        print('Y_train_literal[i]:', le.fit(Y_train_literal[i]))
        Y_train.append(le.transform(Y_train_literal[i]))  
        print('le.transform(Y_train_literal[i]):', le.transform(Y_train_literal[i]))
        num_classes = 11
        Y_train[i] = to_categorical(Y_train[i], num_classes).astype('int')  
        Y_test[i] = to_categorical(Y_test[i], num_classes).astype('int')
        i += 1  
        if k_folds == False:
            break

    return X_train, Y_train, X_test, Y_test, le, labels_literal
def save_model(model=None):
    available_options = ['Y', 'N']
    print('\nModel Trained\n\nDo you wish to save model (Y/N)?')
    while True:
        ans = input("").upper()
        if ans in available_options:
            break
        print("Invalid option.")
    if ans == 'Y':
        filename = input("\nModel name: ")
        if not os.path.exists(f"{OUTPUT_MODEL_DIR}"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/")
        model.save(f"{OUTPUT_MODEL_DIR}/{filename}", save_format='tf')  # {filename}.h5", save_format='h5')
        print(f"Model saved in '{OUTPUT_MODEL_DIR}'\n")
    else:
        n=-1

def save_summary(model, filename):
    if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_summaries"):
        os.makedirs(f"{OUTPUT_MODEL_DIR}/models_summaries/")
    with open(f'{OUTPUT_MODEL_DIR}/summary_{filename}.txt', 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved in '{OUTPUT_MODEL_DIR}/models_summaries/'\n")

# ------------------ swin transformer ------------------------------------------
patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 56  # 32  # Initial image size
input_shape = (56, 56, 3)  # （32，32，3）
num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]
learning_rate = 1e-3
batch_size = 128
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1
"""
## Helper functions
We create two helper functions to help us get a sequence of
patches from the image, merge patches, and apply dropout.
"""
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows
def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x
class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential( 
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)
# ------------------ swin transformer --------------------------------------


#---------------- data preprocessiong -------------------
x_train, y_train, x_test, y_test, le, labels_literal = process_data_VI_Images(k_folds=True) 

#---------------------- make coarse 2 labels --------------------------

parent_f = {
  0:3, 2:3, 5:3, 6:3, 9:3,
  4:2, 8:2,
  3:1, 10:1,
  1:0, 7:0
}

y_c2_train = []
y_c2_test = []
for i in range(len(y_train)):
    y_c2_train.append(np.zeros((y_train[i].shape[0], num_c_2)))
    y_c2_test.append(np.zeros((y_test[i].shape[0], num_c_2)))

for i in range(len(y_train)):
    for j in range(y_c2_train[i].shape[0]):
      y_c2_train[i][j, parent_f[np.argmax(y_train[i][j])]] = 1 
for i in range(len(y_train)):
    for j in range(y_c2_test[i].shape[0]):
      y_c2_test[i][j, parent_f[np.argmax(y_test[i][j])]] = 1 
parent_c2 = {
  0:0, 1:0,
  2:1, 3:1
}

y_c1_train = []
y_c1_test = []
for i in range(len(y_train)):
    y_c1_train.append(np.zeros((y_c2_train[i].shape[0], num_c_1)))
    y_c1_test.append(np.zeros((y_c2_test[i].shape[0], num_c_1)))

for i in range(len(y_test)):
    for j in range(y_c1_train[i].shape[0]):
      y_c1_train[i][j, parent_c2[np.argmax(y_c2_train[i][j])]] = 1
for i in range(len(y_test)):
    for j in range(y_c1_test[i].shape[0]):
      y_c1_test[i][j, parent_c2[np.argmax(y_c2_test[i][j])]] = 1

def get_net_model(alpha, beta, gamma):
    img_input = Input(shape=input_shape, name='input')
    # -----------------swin transformer-------------------------
    img_input = layers.RandomCrop(image_dimension, image_dimension)(img_input) 
    img_input = layers.RandomRotation(0.2)(img_input)
    y_input = layers.RandomFlip("horizontal_and_vertical")(img_input) 
    y_1 = PatchExtract(patch_size)(y_input)
    y = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(y_1)
    print('y.shape:', y)  
    ST1 = SwinTransformer(
        dim=embed_dim,  
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads, 
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp, 
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(y)
    ST2 = SwinTransformer(
        dim=embed_dim, 
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,  
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,  
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(ST1)
    # -----------------swin transformer-------------------------

    # --- block 1 ---
    y_1 = Reshape((56, 56, 3))(y_1)
    x = Conv2D(12, (3, 3), activation='relu', padding='same', name='block1_conv1')(y_1)
    x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)

    # --- coarse 1 branch ---
    c_1_bch = Flatten(name='c1_flatten')(x)
    c_1_bch = Dense(32, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_pred = Dense(num_c_1, activation='softmax', name='c1_p')(c_1_bch)

    # --- block 2 ---
    x = Conv2D(28, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)

    ST1 = Reshape((56, 56, 16))(ST1)
    ST1_brunch = ST1
    ST1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv3')(ST1)
    ST1 = BatchNormalization()(ST1)

    # --- coarse 2 branch ---
    x_out = Add()([ST1, x])
    c_2_bch = Flatten(name='c2_flatten')(x_out)
    c_2_bch = Dense(48, activation='relu', name='c2_fc_cifar10_1')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.2)(c_2_bch)
    c_2_pred = Dense(num_c_2, activation='softmax', name='c2_p')(c_2_bch)

    # --- block 3 ---
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_out)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv4')(x)
    x = BatchNormalization()(x)

    # --- block 4 ---
    # --- fine block ---

    ST2 = Reshape((28, 28, 64))(ST2)

    # shuffleNet---------------------------------
    ST2_left = DepthwiseConv2D(kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same')(ST2)
    # print('ST2_left:', ST2_left)
    ST2_left = BatchNormalization()(ST2_left)
    ST2_left = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu')(ST2_left)
    ST2_left = BatchNormalization()(ST2_left)
    # ST1_bap = GlobalAveragePooling2D()(ST1)
    # # result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)
    # ST1_left = multiply([ST1_left, ST1_bap])
    # print('ST2_after_multiply:',ST1_left)  # shape=(None, 56, 56, 8)
    ST2_left = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(ST2_left)

    ST2_right = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu')(ST2)
    ST2_right = BatchNormalization()(ST2_right)
    ST2_right = DepthwiseConv2D(kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same')(ST2_right)
    # print('ST2_right:', ST2_right)  # shape=(None, 56, 56, 16)
    ST2_right = BatchNormalization()(ST2_right)
    ST2_right = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(ST2_right)
    ST2_right = BatchNormalization()(ST2_right)
    ST2 = Add()([ST2_left, ST2_right])
    ST2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(ST2)
    # shuffleNet---------------------------------

    ST2 = Add()([ST2, x])
    ST2 = Flatten(name='ST2_flatten')(ST2)
    ST2 = Dense(128, activation='relu', name='fine_fc1')(ST2)
    ST2 = Dropout(0.3)(ST2)
    output = layers.Dense(units=NUM_CATEGORIES, activation='softmax', name="last_layer")(ST2) 

    model = Model(inputs=img_input, outputs=[c_1_pred, c_2_pred, output], name='medium_dynamic')
    sgd = SGD(lr=0.003, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  loss_weights=[alpha, beta, gamma],
                  # optimizer=keras.optimizers.Adadelta(),
                  metrics=['acc'])
    return model
#----------------------- model definition ---------------------------
alpha = K.variable(value=0.98, dtype="float32", name="alpha") 
beta = K.variable(value=0.01, dtype="float32", name="beta")
gamma = K.variable(value=0.01, dtype="float32", name="gamma") 

#----------------------- compile and fit ---------------------------
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
change_lw = LossWeightsModifier(alpha, beta, gamma)
cbks = [change_lr, tb_cb, change_lw]
history = []
scores = []
f1_total = []
accuracy_total = []
zl_total = []
mcc_total = []
for i in range(len(x_train)):
    model = get_net_model(alpha, beta, gamma)
    history.append(model.fit(x_train[i], [y_c1_train[i], y_c2_train[i], y_train[i]],
                            batch_size=128,  # 128
                            epochs=epochs,
                            verbose=1,
                            callbacks=cbks,
                            validation_data=(x_test[i], [y_c1_test[i], y_c2_test[i], y_test[i]])))
    # scores.append(model.evaluate(x_test[i], [y_c1_test[i], y_c2_test[i], y_test[i]], verbose=0))
    # A problem has occured when I try to save the best model to conduct the predict, hence the final results are lower than theory(this can be seen by observing the last layer accuracy fluctuating in training). 
    # Our paper use the mean value of 3 experiment results as the final results.
    prediction = model.predict(x_test[i])[2]
    y_true = np.argmax(y_test[i], axis=1)  
    prediction = np.argmax(prediction, axis=1) 
    accuracy = accuracy_score(y_true, prediction)
    precision = precision_score(y_true, prediction, average='macro')
    recall = recall_score(y_true, prediction, average='macro')
    f1 = f1_score(y_true, prediction, average='macro')
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Error rate: {:.2f}%'.format((1 - accuracy) * 100))
    print('Precision: {:.2f}%'.format(precision * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('F1: {:.2f}%'.format(f1 * 100))
    accuracy_total.append(f1)
    f1_total.append(f1)
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_true, prediction)
    mcc_total.append(mcc)

f1_mean =0.0
mcc_mean=0.0
for i,j,k in zip(f1_total,mcc_total,zl_total):
    f1_mean = f1_mean + float(i)
    mcc_mean = mcc_mean + float(j)
f1_mean = f1_mean/10
mcc_mean =mcc_mean/10
print('f1_mean:',f1_mean)
print('mcc_mean:',mcc_mean)
with open('2014_2017_model_K.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['F1_mean', 'mcc_mean'])
    writer.writerow([f1_mean, mcc_mean])
def plot_training_results(model, history, epochs, filename):
    plt.figure(figsize=(16, 8))
    subtitles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    j = 1
    acc = []
    loss = []
    for i in range(len(history)):
        acc.append(history[i].history['acc'])
        loss.append(history[i].history['loss'])
        epochs_range = range(epochs)
        plt.subplot(len(history), 2, j)
        plt.title(subtitles[j - 1], fontsize=10, pad=10)
        plt.plot(epochs_range, acc[i], label=f'Training Accuracy {i + 1}')
        plt.legend(loc='lower right')
        j += 1
        plt.subplot(len(history), 2, j)
        plt.title(subtitles[j - 1], fontsize=10, pad=10)
        plt.plot(epochs_range, loss[i], label=f'Training Loss {i + 1}')
        plt.legend(loc='upper right')
        j += 1
    plt.tight_layout()
    if not os.path.exists(f"{OUTPUT_MODEL_DIR}/training_history"):
        os.makedirs(f"{OUTPUT_MODEL_DIR}/training_history/")
    plt.savefig(f"{OUTPUT_MODEL_DIR}/training_history/{filename}.png", dpi=128)

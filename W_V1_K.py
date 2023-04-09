import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input,Reshape,Add
from keras.layers import Dense, Flatten, Dropout, Layer, GlobalAveragePooling1D, Input, RandomCrop, RandomFlip, Embedding,LayerNormalization,Activation
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers import BatchNormalization
from keras.utils import get_file
from keras import backend as K
from scipy.io import loadmat
import seaborn as sn
import pandas as pd
from keras import layers, models
from keras.losses import categorical_crossentropy
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
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
EPOCHS = 60 # 100  # ---
EPOCHS_LENET = 10
IMG_WIDTH = 56 # 128
IMG_HEIGHT = 56 # 128
NUM_CATEGORIES = 55
INPUT_IMAGE_DIR = "images/VI_images_test/Whited/"  
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
num_c_2 = 5
#--- fine classes ---
num_classes = 55

batch_size = 128
epochs = 85 # 60

#--- file paths ---
log_filepath = './tb_log_medium_dynamic/'
weights_store_filepath = './models/'   # B_CNN
train_id = '1'
model_name = 'BCNN_SVHN'+train_id
model_path = os.path.join(weights_store_filepath, model_name)

def scheduler(epoch):
  learning_rate_init = 0.003
  if epoch > 50:
    learning_rate_init = 0.002 # 0.0005   # 调整成 0.001？
  if epoch > 70:
    learning_rate_init = 0.001 # 0.0001   # 调整成 0.0007？
  return learning_rate_init

class LossWeightsModifier(tf.keras.callbacks.Callback):
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 8:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.8)
      K.set_value(self.gamma, 0.1)
    if epoch == 18:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.2)
      K.set_value(self.gamma, 0.7)
    if epoch == 28:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 0)
      K.set_value(self.gamma, 1)
def import_model():
    print('请输入训练好的的模型进行测试：')
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
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
            img = np.array(img)  
            images.append(img)
            labels.append(str(category)) 
        labels_literal.append(str(category))  
    return (images, labels), labels_literal
def load_data_1(data_dir):
    Class = ['AC', 'AirPump', 'BenchGrinder', 'CableModem', 'CableReceiver', 'CFL', 'Charger',
             'CoffeeMachine', 'DeepFryer', 'DesktopPC', 'Desoldering', 'DrillingMachine', 'Fan', 'FanHeater',
             'FlatIron', 'Fridge','GameConsole','GuitarAmp','HairDryer','Halogen','HalogenFluter','Heater','HIFI',
             'Iron','JigSaw','JuiceMaker','Kettle','KitchenHood','Laptop','LaserPrinter','LEDLight','LightBulb',
             'Massage','Microwave','Mixer','Monitor','MosquitoRepellent','MultiTool','NetworkSwitch','PowerSupply',
             'Projector','RiceCooker','SandwichMaker','SewingMachine','ShoeWarmer','Shredder','SolderingIron',
             'Stove','Toaster','Treadmill','TV','VacuumCleaner','WashingMachine','WaterHeater','WaterPump']
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
    X = X/255  # 对图片进行归一化
    print('X[0]-----after:', X[0])
    Y = np.array(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True)  #-------------------------------------

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
        # print('X_train[i].shape[0]:', X_train[i].shape[0])  # 5484
        # print('Y_test_literal[i]:', le.fit(Y_test_literal[i]))
        Y_test.append(le.transform(Y_test_literal[i]))  
        # print('le.transform(Y_test_literal[i]):', le.transform(Y_test_literal[i]))
        le.fit(Y_train_literal[i]) 
        print('Y_train_literal[i]:', le.fit(Y_train_literal[i]))
        Y_train.append(le.transform(Y_train_literal[i]))  #
        print('le.transform(Y_train_literal[i]):', le.transform(Y_train_literal[i]))
        num_classes = 55
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
def preprocess_test_VI_Images():
    (images, labels), labels_literal = load_data(INPUT_IMAGE_DIR)

    for dirname, _, filenames in os.walk('images/VI_images_test/0318_final_RGB'):
        for filename in filenames:
            os.path.join(dirname, filename)

    DATA_DIR = 'images/VI_images_test/0318_final_RGB'
    Class = ['Air Conditioner', 'Blender', 'Coffee maker', 'Compact Fluorescent Lamp', 'Fan', 'Fridge', 'Hair Iron',
             'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Laptop', 'Microwave', 'Soldering Iron', 'Vacuum',
             'Washing Machine', 'Water kettle']
    IMG_SZ = 56

    trn_data = []

    def create_training_data():
        for cat in Class:
            path = os.path.join(DATA_DIR, cat)
            class_num = Class.index(cat)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))
                    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)  # 若是3通道的图片，则会转成单通道图片
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

    raw_images, raw_labels = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)
    raw_images = raw_images / 255.0
    print('raw_labels:', raw_labels)
    print('raw_labels.shape', raw_labels.shape)
    print('raw_labels[4]:', raw_labels[4])
    print('raw_labels[200]:', raw_labels[200])
    # # one_hot编码
    one_hot_labels = to_categorical(raw_labels)
    X_train, X_test, y_train, y_test = train_test_split(raw_images, one_hot_labels, test_size=0.2, stratify=one_hot_labels)
    return X_train, X_test, y_train, y_test, labels_literal
def save_summary(model, filename):
    if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_summaries"):
        os.makedirs(f"{OUTPUT_MODEL_DIR}/models_summaries/")
    with open(f'{OUTPUT_MODEL_DIR}/summary_{filename}.txt', 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved in '{OUTPUT_MODEL_DIR}/models_summaries/'\n")

# ------------------ swim transformer ------------------------------------------
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
"""
## Window based multi-head self-attention
Usually Transformers perform global self-attention, where the relationships between
a token and all other tokens are computed. The global computation leads to quadratic
complexity with respect to the number of tokens. Here, as the [original paper](https://arxiv.org/abs/2103.14030)
suggests, we compute self-attention within local windows, in a non-overlapping manner.
Global self-attention leads to quadratic computational complexity in the number of patches,
whereas window-based self-attention leads to linear complexity and is easily scalable.
"""
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
"""
## The complete Swin Transformer model
Finally, we put together the complete Swin Transformer by replacing the standard multi-head
attention (MHA) with shifted windows attention. As suggested in the
original paper, we create a model comprising of a shifted window-based MHA
layer, followed by a 2-layer MLP with GELU nonlinearity in between, applying
`LayerNormalization` before each MSA layer and each MLP, and a residual
connection after each of these layers.
Notice that we only create a simple MLP with 2 Dense and
2 Dropout layers. Often you will see models using ResNet-50 as the MLP which is
quite standard in the literature. However in this paper the authors use a
2-layer MLP with GELU nonlinearity in between.
"""
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

        self.mlp = keras.Sequential(  # --------------------------------------------------
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
"""
## Model training and evaluation
### Extract and embed patches
We first create 3 layers to help us extract, embed and merge patches from the
images on top of which we will later use the Swin Transformer class we built.
"""
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
# ------------------ swim transformer --------------------------------------

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train.shape:',x_train.shape)  # (50000, 32, 32, 3)
# print('y_train.shape:',y_train.shape)  # (50000, 1)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# print('x_train:', len(x_train))
#---------------- data preprocessiong -------------------
# x_train = (x_train-np.mean(x_train)) / np.std(x_train)
# x_test = (x_test-np.mean(x_test)) / np.std(x_test)
# print('x_train.shape:',x_train.shape)  # (50000, 32, 32, 3)
# print('x_test.shape:',x_test.shape)  # (10000, 32, 32, 3)

# 加载VI轨迹数据集--------------------
# x_train, x_test, y_train, y_test, labels_literal = preprocess_test_VI_Images()  # 第3步其实已经分出训练和测试集了
# print('x_test的shape：', x_test.shape)  # (376, 56, 56, 3)
# print('y_test的shape：', y_test.shape)  # (376, 16)
# x_train = (x_train-np.mean(x_train)) / np.std(x_train)
# x_test = (x_test-np.mean(x_test)) / np.std(x_test)
# print('x_train.shape:', x_train.shape)  # (1500, 56, 56, 3)
# print('x_test.shape:', x_test.shape)  # (376, 56, 56, 3)
# print('y_train.shape:', y_train.shape)  # (1500, 16)


x_train, y_train, x_test, y_test, le, labels_literal = process_data_VI_Images(k_folds=True)  # K 对val数值影响很大
print('len(x_train):', len(x_train)) # 4
print('len(x_test):', len(x_test)) # 4
print('len(x_test[0]):', len(x_test[0]))  # 图片数量可能会引起问题
print('len(x_test[1]):', len(x_test[1]))  # 
print('len(x_test[2]):', len(x_test[2]))  # 
print('len(x_test[3]):', len(x_test[3]))  #
print('len(x_test[4]):', len(x_test[4]))  #
print('y_train[1][400]:', y_train[1][400])  # [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
print('y_train[1]:', y_train[1])
print('(x_train):',(x_train))
# x_train list:4 (1407,56,56,3)
# x_test list:4 (469,56,56,3)
# y_train list:4 (1407,16)
# y_test list:4 (469,16)
#---------------------- make coarse 2 labels --------------------------
# parent_f = {
#   2:3, 3:5, 5:5,
#   1:2, 7:6, 4:6,
#   0:0, 6:4, 8:1, 9:2
# }
parent_f = {
  5:4, 11:4, 14:4, 19:4,24:4, 25:4, 31:4, 37:4, 44:4,45:4, 51:4,
  30:3, 33:3,34:3, 35:3,36:3, 43:3,46:3, 48:3,50:3, 54:3,
  0:2, 2:2, 9:2, 10:2,12:2, 15:2, 17:2, 27:2, 39:2, 52:2,
  3:1, 4:1, 6:1, 16:1,22:1, 28:1, 29:1, 32:1, 38:1,40:1, 49:1,
  1:0, 7:0, 8:0, 13:0, 18:0, 20:0, 21:0, 23:0, 26:0, 41:0, 42:0, 47:0, 53:0
}

# y_c2_train = np.zeros((len(y_train), y_train[0].shape[0], num_c_2)).astype("float32")
# y_c2_test = np.zeros((len(y_train), y_test[0].shape[0], num_c_2)).astype("float32")
y_c2_train = [np.zeros((y_train[0].shape[0], num_c_2)) for i in range(len(y_train))]
y_c2_test = [np.zeros((y_test[0].shape[0], num_c_2)) for i in range(len(y_test))]
print('y_train.shape[0]:',y_train[0].shape[0]) #  1407
print('y_test.shape[0]:',y_test[0].shape[0]) # 469
print('y_c2_train.shape:', y_c2_train[0].shape)  # y_c2_train.shape: (1407, 4)
print('y_c2_test.shape:', y_c2_test[0].shape)  # (469, 4)
print('y_c2_test.shape[0]:', y_c2_test[0].shape[0])  # 469
for i in range(len(y_train)):
    for j in range(y_c2_train[i].shape[0]):
      y_c2_train[i][j, parent_f[np.argmax(y_train[i][j])]] = 1  # 图片数量可能会引起问题
      # print("parent_f[np.argmax(y_c2_train[i])]:", parent_f[np.argmax(y_c2_train[i])])
for i in range(len(y_train)):
    for j in range(y_c2_test[i].shape[0]):
      y_c2_test[i][j, parent_f[np.argmax(y_test[i][j])]] = 1  # ???  图片数量可能会引起问题
      # print("parent_f[np.argmax(y_c2_train[i])]:", parent_f[np.argmax(y_c2_train[i])])
# for i in range(y_c2_test.shape[0]):
#   y_c2_test[i][parent_f[np.argmax(y_test[i])]] = 1.0
# print('y_c2_train.shape:', y_c2_train.shape)  # y_c2_train.shape: (50000, 7)
# print('y_c2_test.shape:', y_c2_test.shape)  # y_c2_test.shape: (10000, 7)
#---------------------- make coarse 1 labels --------------------------
# parent_c2 = {
#   0:0, 1:0, 2:0,
#   3:1, 4:1, 5:1, 6:1
# }
parent_c2 = {
  0:0,1:0,2:0,
  3:1, 4:1
}
# y_c1_train = np.zeros((y_c2_train.shape[0], num_c_1)).astype("float32")
# y_c1_test = np.zeros((y_c2_test.shape[0], num_c_1)).astype("float32")
y_c1_train = [np.zeros((y_c2_train[0].shape[0], num_c_1)) for i in range(len(y_train))]
y_c1_test = [np.zeros((y_c2_test[0].shape[0], num_c_1)) for i in range(len(y_test))]
for i in range(len(y_test)):
    for j in range(y_c1_train[i].shape[0]):
      y_c1_train[i][j, parent_c2[np.argmax(y_c2_train[i][j])]] = 1
      # print("parent_f[np.argmax(y_c2_train[i])]:", parent_f[np.argmax(y_c2_train[i])])
for i in range(len(y_test)):
    for j in range(y_c1_test[i].shape[0]):
      y_c1_test[i][j, parent_c2[np.argmax(y_c2_test[i][j])]] = 1
      # print("parent_c2[np.argmax(y_c2_test[i][j])]:", parent_c2[np.argmax(y_c2_test[i][j])])

# for i in range(y_c1_train.shape[0]):
#   y_c1_train[i][parent_c2[np.argmax(y_c2_train[i])]] = 1.0
#   print("parent_c2[np.argmax(y_c2_train[i])]:", parent_c2[np.argmax(y_c2_train[i])])
# for i in range(y_c1_test.shape[0]):
#   y_c1_test[i][parent_c2[np.argmax(y_c2_test[i])]] = 1.0

# print('y_c1_train.shape:', y_c1_train.shape)  # y_c1_train.shape: (50000, 2)
# print('y_c1_test.shape:', y_c1_test.shape)  # y_c1_test.shape: (10000, 2)
Resnet_model = import_model()
Resnet_model = tf.keras.models.Sequential(Resnet_model.layers[:-1])  # 删掉最后一层
for layer in Resnet_model.layers:
    layer.trainable = False

# VI_model = get_model(x_train)  # 前面段的网络模型
img_input = Input(shape=input_shape, name='input')

# -----------------swim transformer-------------------------
y = layers.RandomCrop(image_dimension, image_dimension)(img_input)
print('x.shape:',y) # (None, 32,32, 3)
y = layers.RandomFlip("horizontal")(y)
print('x.shape:',y) # (None, 32,32, 3)
y = PatchExtract(patch_size)(y)
print('x.shape:',y) # (None, 256, 12)
y = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(y)
print('x.shape:',y) # (None, 784, 64)
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
print('x.shape:',y) # (None, 784, 64)
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
print('x.shape:',ST2) # (None, 784, 64)
ST2 = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(ST2)
print('x.shape:',ST2) # (None, 196, 128)
y = layers.GlobalAveragePooling1D()(ST2)
print('x.shape:',y) # (None, 128)
# -----------------swim transformer-------------------------

# --- block 1 ---
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
print('x.shape:',x.shape)  # (None, 56, 56, 16)
x = BatchNormalization()(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# --- block 2 ---
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
print('block 2.x.shape:', x.shape)  # block 2.x.shape: (None, 8, 8, 128)

# --- coarse 1 branch ---
c_1_bch = Flatten(name='c1_flatten')(x)
c_1_bch = Dense(32, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
c_1_bch = BatchNormalization()(c_1_bch)
c_1_bch = Dropout(0.4)(c_1_bch)
c_1_bch = Dense(32, activation='relu', name='c1_fc2')(c_1_bch)
c_1_bch = BatchNormalization()(c_1_bch)
c_1_bch = Dropout(0.4)(c_1_bch)
c_1_pred = Dense(num_c_1, activation='softmax', name='c1_p')(c_1_bch)
print('coarse 1.c_1_pred.shape:', c_1_pred.shape)  # coarse 1.c_1_pred.shape: (None, 2)

# --- block 3 ---
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
# x = BatchNormalization()(x)
x = Conv2D(28, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
print('block 3.x.shape:', x.shape)  # block 3.x.shape: (None, 4, 4, 256)

# --- coarse 2 branch ---
ST1 = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(ST1)
print('ST1.shape:',ST1) # (None, 196, 128)
ST1 = Reshape((56,56,8))(ST1)
print('ST1.shape:',ST1) # (None, 196, 128)
x_out = Add()([ST1, x])
# y = layers.GlobalAveragePooling1D()(ST1)
c_2_bch = Flatten(name='c2_flatten')(x_out)
c_2_bch = Dense(48, activation='relu', name='c2_fc_cifar10_1')(c_2_bch)
c_2_bch = BatchNormalization()(c_2_bch)
c_2_bch = Dropout(0.3)(c_2_bch)
c_2_bch = Dense(32, activation='relu', name='c2_fc2')(c_2_bch)
c_2_bch = BatchNormalization()(c_2_bch)
c_2_bch = Dropout(0.2)(c_2_bch)
c_2_pred = Dense(num_c_2, activation='softmax', name='c2_p')(c_2_bch)
print('coarse 2.c_2_pred.shape:', c_2_pred.shape)  # coarse 2.c_2_pred.shape: (None, 7)

# --- block 4 ---
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block4_conv1')(x_out)
# x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
# x = BatchNormalization()(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
print('block 4.x.shape:', x.shape)  # block 4.x.shape: (None, 2, 2, 512)

# --- fine block ---
x = layers.MaxPool2D(pool_size=(2, 2), padding='valid', name='Pooling_1')(x)
x = layers.Conv2D(filters=16, kernel_size=(1, 1), padding='valid', activation='relu')(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='valid', activation='relu')(x)
x = layers.ZeroPadding2D((1, 1))(x)
print('block 4.x.shape-------:', x.shape)

print('ST2.shape:',ST2) # (None, 196, 128)
ST2 = Reshape((56,56,8))(ST2)
ST2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='f_conv1')(ST2)
ST2 = Reshape((32, 32, 49))(ST2)
ST2 = Conv2D(3, (3, 3), activation='relu', padding='same', name='f_conv2')(ST2)
ST2 = Add()([ST2, x])
# x = Conv2D(3, (3, 3), activation='relu', padding='same', name='fine_block')(x)
# x = BatchNormalization()(x)
print('fine block.x.shape:', ST2.shape)  # fine block.x.shape: (None, 56,56,3)
VI_model = tf.keras.Model(inputs=img_input, outputs=ST2, name="ResNet Extension")
output_vi = VI_model.layers[-1].output  # 输出最后一层  # 满足（32，32，3）即可
print("output_vi.shape:", output_vi.shape)
print("VI_model.layers[-2].output.shape:", VI_model.layers[-2].output.shape)
output = Resnet_model(inputs=output_vi)  # 前面模型的最后一层的输出作为另一个模型的输入
output = layers.Dense(units=NUM_CATEGORIES, activation='softmax', name="last_layer")(output)  # 模型的最后一层
# complete_model = tf.keras.Model(inputs=VI_model.input, outputs=output)
model = Model(inputs=VI_model.input, outputs=[c_1_pred, c_2_pred, output], name='medium_dynamic')
model.summary()
#----------------------- model definition ---------------------------
alpha = K.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

#----------------------- compile and fit ---------------------------
sgd = SGD(lr=0.003, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              loss_weights=[alpha, beta, gamma],
              # optimizer=keras.optimizers.Adadelta(),
              metrics=['acc'])

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
change_lw = LossWeightsModifier(alpha, beta, gamma)
cbks = [change_lr, tb_cb, change_lw]
history = []
scores = []
f1_total = []
accuracy_total = []
for i in range(len(x_train)):
    history.append(model.fit(x_train[i], [y_c1_train[i], y_c2_train[i], y_train[i]],
                            batch_size=64,  # 128
                            epochs=epochs,
                            verbose=1,
                            callbacks=cbks,
                            validation_data=(x_test[i], [y_c1_test[i], y_c2_test[i], y_test[i]])))
    scores.append(model.evaluate(x_test[i], [y_c1_test[i], y_c2_test[i], y_test[i]], verbose=0))
    prediction = model.predict(x_test[i])[2]
    y_true = np.argmax(y_test[i], axis=1)  # ---orignal  y_test[0]
    # y_true = np.argmax(y_test, axis=1)
    # prediction = VI_resnet_model.predict(x_test[0])
    prediction = np.argmax(prediction, axis=1)  # 取最大值的索引（0-15）作为预测标签
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
# plot_training_results(model, history, EPOCHS, filename='history_VI_lenet_')
# model.compile(loss='categorical_crossentropy',
#             # optimizer=keras.optimizers.Adadelta(),
#             optimizer=sgd,
#             metrics=['accuracy'])
# model.save(model_path, save_format='tf')  # 保存为tf格式，便于后续迁移调用   {filename}.h5", save_format='h5')


# VI_resnet_model = import_model()
prediction_uni = []
label_uni = []
# x_train, y_train, x_test, y_test, le, labels_literal = process_data_VI_Images(k_folds=False)  # ---orignal
# x_train, x_test, y_train, y_test ,labels_literal= preprocess_test_VI_Images()
# print('x_test[0]的shape：', x_test.shape)  # (1828, 128, 128, 1)
# print('y_test[0]的shape：', y_test.shape)  # (1828, 16)
# prediction = VI_resnet_model.predict(x_test[0])  # ndarry (1828,16)得到的是概率值
# x_test = x_test.reshape(1463,128,128,1)
# prediction = VI_resnet_model.predict(x_test[0])[2]  # ---orignal  x_test[0] # ndarry (1828,16)得到的是概率值
# print('x_test的shape：', x_test.shape)  # (1828, 128, 128, 1)
# print('prediction的shape：', prediction.shape)  # (1828, 16)
# print('y_test的shape：', y_test.shape)  # (1828, 16)
# print('y_test的shape：', y_test.shape)  # 'list' object has no attribute 'shape'
# for j in range(len(prediction)):  # -------
#     prediction_uni.append(np.argmax([prediction[j]]))  # 取出a中元素最大值所对应的索引
#     label_uni.append(np.argmax([y_test[0][j]]))  # orignal: y_test[0][j]


# confusionMatrix = tf.math.confusion_matrix(labels=label_uni, predictions=prediction_uni, num_classes=16)
# df_cm = pd.DataFrame(confusionMatrix, index=[i for i in labels_literal],
#                      columns=[i for i in labels_literal])
# plt.figure(figsize=(12, 8))
# sn.heatmap(df_cm, annot=True, fmt='g')
# plt.tight_layout()
# if not os.path.exists(f"{SAVE_DIR}/confusion_matrix"):
#     os.makedirs(f"{SAVE_DIR}/confusion_matrix/")
# plt.savefig(f"{SAVE_DIR}/confusion_matrix/confusion_matrix_03.png", dpi=128)
# print(f"Confusion Matrix saved in '{SAVE_DIR}/confusion_matrix/'\n")

# ----metrics----

# prediction = model.predict(x_test[0])[2]
# y_true = np.argmax(y_test[0], axis=1)  # ---orignal  y_test[0]
# # y_true = np.argmax(y_test, axis=1)
# # prediction = VI_resnet_model.predict(x_test[0])
# prediction = np.argmax(prediction, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy = accuracy_score(y_true, prediction)
# precision = precision_score(y_true, prediction, average='macro')
# recall = recall_score(y_true, prediction, average='macro')
# f1 = f1_score(y_true, prediction, average='macro')
# print('Accuracy: {:.2f}%'.format(accuracy * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy) * 100))
# print('Precision: {:.2f}%'.format(precision * 100))
# print('Recall: {:.2f}%'.format(recall * 100))
# print('F1: {:.2f}%'.format(f1 * 100))
#
# prediction_1 = model.predict(x_test[1])[2]
# y_true_1 = np.argmax(y_test[1], axis=1)  # ---orignal  y_test[0]
# prediction_1 = np.argmax(prediction_1, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy_1 = accuracy_score(y_true_1, prediction_1)
# precision_1 = precision_score(y_true_1, prediction_1, average='macro')
# recall_1 = recall_score(y_true_1, prediction_1, average='macro')
# f1_1 = f1_score(y_true_1, prediction_1, average='macro')
# print('accuracy_1: {:.2f}%'.format(accuracy_1 * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy_1) * 100))
# print('Precision_1: {:.2f}%'.format(precision_1 * 100))
# print('Recall_1: {:.2f}%'.format(recall_1 * 100))
# print('F1_1: {:.2f}%'.format(f1_1 * 100))
#
# prediction_2 = model.predict(x_test[2])[2]
# y_true_2 = np.argmax(y_test[2], axis=1)  # ---orignal  y_test[0]
# prediction_2 = np.argmax(prediction_2, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy_2 = accuracy_score(y_true_2, prediction_2)
# precision_2 = precision_score(y_true_2, prediction_2, average='macro')
# recall_2 = recall_score(y_true_2, prediction_1, average='macro')
# f1_2 = f1_score(y_true_2, prediction_2, average='macro')
# print('accuracy_2: {:.2f}%'.format(accuracy_2 * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy_2) * 100))
# print('Precision_2: {:.2f}%'.format(precision_2 * 100))
# print('Recall_2: {:.2f}%'.format(recall_2 * 100))
# print('F1_2: {:.2f}%'.format(f1_2 * 100))
#
# prediction_3 = model.predict(x_test[3])[2]
# y_true_3 = np.argmax(y_test[3], axis=1)  # ---orignal  y_test[0]
# prediction_3 = np.argmax(prediction_3, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy_3 = accuracy_score(y_true_3, prediction_3)
# precision_3 = precision_score(y_true_3, prediction_3, average='macro')
# recall_3 = recall_score(y_true_3, prediction_3, average='macro')
# f1_3 = f1_score(y_true_3, prediction_3, average='macro')
# print('accuracy_3: {:.2f}%'.format(accuracy_3 * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy_3) * 100))
# print('Precision_3: {:.2f}%'.format(precision_3 * 100))
# print('Recall_3: {:.2f}%'.format(recall_3 * 100))
# print('F1_3: {:.2f}%'.format(f1_3 * 100))
#
#
# x_train, y_train, x_test, y_test, le, labels_literal = process_data_VI_Images(k_folds=False)  # ---orignal
# prediction = model.predict(x_test[0])[2]
# y_true = np.argmax(y_test[0], axis=1)  # ---orignal  y_test[0]
# # y_true = np.argmax(y_test, axis=1)
# # prediction = VI_resnet_model.predict(x_test[0])
# prediction = np.argmax(prediction, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy = accuracy_score(y_true, prediction)
# precision = precision_score(y_true, prediction, average='macro')
# recall = recall_score(y_true, prediction, average='macro')
# f1 = f1_score(y_true, prediction, average='macro')
# print('Accuracy: {:.2f}%'.format(accuracy * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy) * 100))
# print('Precision: {:.2f}%'.format(precision * 100))
# print('Recall: {:.2f}%'.format(recall * 100))
# print('F1: {:.2f}%'.format(f1 * 100))

# # ------------------begin---------------------
# model.fit(x_train, [y_c1_train, y_c2_train, y_train],
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           callbacks=cbks,
#           validation_data=(x_test, [y_c1_test, y_c2_test, y_test]))
# #---------------------------------------------------------------------------------
# # The following compile() is just a behavior to make sure this model can be saved.
# # We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
# #---------------------------------------------------------------------------------
# model.compile(loss='categorical_crossentropy',
#             # optimizer=keras.optimizers.Adadelta(),
#             optimizer=sgd,
#             metrics=['accuracy'])
#
# # score = model.evaluate(x_test, [y_c1_test, y_c2_test, y_test], verbose=0)
# # 使用 evaluate 函数计算每个输出层的指标
# loss, c1_loss, c2_loss, y_loss, c1_acc, c2_acc, y_acc = model.evaluate(x_test, [y_c1_test, y_c2_test, y_test], verbose=0)
# print('y_acc is: ', y_acc)
# # 计算每个输出层的 F1-score
# from sklearn.metrics import f1_score
# import numpy as np
#
# c1_f1 = f1_score(np.argmax(y_c1_test, axis=1), np.argmax(model.predict(x_test)[0], axis=1), average='weighted')
# c2_f1 = f1_score(np.argmax(y_c2_test, axis=1), np.argmax(model.predict(x_test)[1], axis=1), average='weighted')
# y_f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test)[2], axis=1), average='weighted')
# print('y_f1 is: ', y_f1)
# # model.save(model_path)
# model.save(model_path, save_format='tf')  # 保存为tf格式，便于后续迁移调用   {filename}.h5", save_format='h5')
# print(f"Model saved in '{model_path}'\n")
#
# prediction = model.predict(x_test)[2]  # ---orignal  x_test[0] # ndarry (1828,16)得到的是概率值
# print('prediction的shape：', prediction[0].shape)  # (376, 2)
# # print('y_test的shape：', y_test.shape)  # 'list' object has no attribute 'shape'
# prediction_uni = []
# label_uni = []
# for j in range(len(prediction)):  # -------
#     prediction_uni.append(np.argmax([prediction[j]]))  # 取出a中元素最大值所对应的索引
#     label_uni.append(np.argmax([y_test[j]]))  # orignal: y_test[0][j]
# confusionMatrix = tf.math.confusion_matrix(labels=label_uni, predictions=prediction_uni, num_classes=16)
# df_cm = pd.DataFrame(confusionMatrix, index=[i for i in labels_literal],
#                      columns=[i for i in labels_literal])
# plt.figure(figsize=(12, 8))
# sn.heatmap(df_cm, annot=True, fmt='g')
# plt.tight_layout()
#
# # ----metrics----
# y_true = np.argmax(y_test, axis=1)
# print('y_test[0]:', y_test[0])
# print('y_test[1]:', y_test[1])
# prediction = np.argmax(prediction, axis=1)  # 取最大值的索引（0-15）作为预测标签
# accuracy = accuracy_score(y_true, prediction)
# print('Accuracy: {:.2f}%'.format(accuracy * 100))
# print('Error rate: {:.2f}%'.format((1 - accuracy) * 100))
# precision = precision_score(y_true, prediction, average='macro')
# recall = recall_score(y_true, prediction, average='macro')
# f1 = f1_score(y_true, prediction, average='weighted')
# print('Precision: {:.2f}%'.format(precision * 100))
# print('Recall: {:.2f}%'.format(recall * 100))
# print('F1: {:.2f}%'.format(f1 * 100))
# # ----metrics----
#
# # 预测测试集的最后一层分类结果
# # y_pred = model.predict(x_test)[2]
# # y_pred = np.argmax(y_pred, axis=1)
# # y_true = np.argmax(y_test, axis=1)
# # # 计算 precision、recall、F1-score
# # # target_names = ['class_{}'.format(i) for i in range(k)]
# # # print(classification_report(y_true, y_pred, target_names=target_names))
# #
# # # 计算 accuracy
# # loss, c1_loss, c2_loss, y_loss, c1_acc, c2_acc, y_acc = model.evaluate(x_test, [y_c1_test, y_c2_test, y_test], verbose=0)
# # print('Accuracy:', y_acc)
# # precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
# # print('Precision:', precision)
# # print('Recall:', recall)
# # print('F1-score:', f1)
# # -----------------end---------------------

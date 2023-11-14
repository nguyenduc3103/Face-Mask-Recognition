import tensorflow as tf
from utils import preprocessing, calculate_features
from keras_vggface.vggface import VGGFace


gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUS: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = VGGFace(model='resnet50', 
                include_top=False, 
                input_shape=(250, 250, 3),
                pooling='avg')

if __name__ == "__main__":
    url = "data"
    calculate_features(url, model)
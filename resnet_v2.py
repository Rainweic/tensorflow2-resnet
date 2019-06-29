import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PreActBlock(keras.Model):
    """
    Pre-activation version of the BasicBlock.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(planes, (3, 3), strides=stride, padding='same', use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, (3,3), strides=stride, padding='same', use_bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = layers.Conv2D(
                self.expansion * planes,
                (1, 1),
                strides=stride,
                use_bias = False
            )

    def call(self, x):
        out = keras.activations.relu(self.bn1(x))
        out = self.conv1(out)
        out = keras.activations.relu(self.bn2(out))
        out = self.conv2(out)

        out = out + x

        return out


    


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense
from tensorflow.keras.layers import Input, Flatten, AveragePooling2D

def weight_layer(inputs,
                 filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu', 
                 batch_normalization=True, 
                 conv_first=True):
    '''
    这里仅仅只是实现残差中的 (V1)Conv->BN->Activation 或者 (V2)BN->Activation->Conv
    args:
        inputs(tensor):     最初输入的图片或者上一层输入的图片
        filters(int):       卷积核的数量
        kernel_size(int):   卷积核的大小
        strides(int):       卷积核移动的距离
        activation(str):    激活函数
        batch_normalization(bool): 是否使用batch_normalization
        conv_first(bool):   Resnet使用版本, resnetv1(True), resnetv2(False)
    '''
    conv = Conv2D(filters, 
                  kernel_size=kernel_size,
                  strides=strides, 
                  padding='same', 
                  kernel_initializer='he_normal',   
                  kernel_regularizer=keras.regularizers.l2(1e-4))
    
    x = inputs
    if conv_first:
        # Resnet v1
        x = conv(x)
        if batch_normalization:     # 可以随意选择是否添加BN层
            x = BatchNormalization()(x)
        if activation is not None:  # 可以随意选择是否添加激活函数层
            x = Activation(activation)(x)
    else:
        # Resnet v2
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    '''
    实现Resnet V1网络

    args:
        input_shape(tube): 输入图像的shape，例如(128, 128, 3)
        depth(int):        网络的深度
        num_classes(int):  分类器输出结果的种类 
    
    return:
        model：            Resnet V1网络模型
    '''
    # depth必须是6n+2 原因不知道，求大佬告知 难道是range(3)的问题？？？
    if (depth - 2) % 6 != 0:
        raise ValueError('网络深度必须是 6n+2！')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = weight_layer(inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2

            y = weight_layer(x, filters=num_filters, strides=strides)
            # 这里只添加了一个卷积层以及一个BN层
            y = weight_layer(y, filters=num_filters, activation=None)
            # 之后相加，纬度不对应，使用size=1的卷积核进行调整，让x的纬度变大以便与y相加（通过num_filters让纬度相同）
            if stack > 0 and res_block == 0:
                x = weight_layer(x, 
                filters=num_filters, 
                kernel_size=1, 
                strides=strides, 
                activation=None, 
                batch_normalization=False)

            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    
    # 添加分类器
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = resnet_v1((64,64,3), 14)
    keras.utils.plot_model(model, "resnet.png", show_shapes=True)
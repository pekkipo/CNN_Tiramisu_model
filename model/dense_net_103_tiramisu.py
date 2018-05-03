'''
Architecture is shown here
https://arxiv.org/pdf/1611.09326.pdf](One
'''

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dropout, Conv2DTranspose, Reshape, Permute
from keras.optimizers import RMSprop
from keras.regularizers import l2
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from keras import Sequential

num_layers = 5
p = 0.2


def dense_layer(num_layers, num_filters, input):
    '''
    :param num_layers: How many convolutional layers in one dense block
    :param num_filters: Number of filters (feature maps) as an output of this layer
    :param input: input
    :return:
    '''
    temp = input
    del input
    for i in range(num_layers):
        batch_norm = BatchNormalization()(temp)
        relu = Activation('relu')(batch_norm)
        conv2D_3_3 = Conv2D(num_filters, (3, 3), use_bias=False, padding='same')(relu)
        dropout = Dropout(p)(conv2D_3_3)
        temp = dropout
    return temp

def transition_down(num_filters, input):
    '''
    :param num_filters:
    :param input: previous layer
    :return:
    '''
    batch_norm = BatchNormalization()(input)
    relu = Activation('relu')(batch_norm)
    conv2D_1_1 = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(relu)
    dropout = Dropout(p)(conv2D_1_1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(dropout)
    return max_pooling

def transition_up(num_filters, input, input_shape):
    return Conv2DTranspose(num_filters,  kernel_size=(3, 3), strides=(2, 2), input_shape=input_shape, padding='same', kernel_regularizer=l2(0))(input)

def create_tiramisu_model(input_shape=(224, 224, 3), num_classes=1):

    '''
    FC-DenseNet103
    growth_rate = 16
    block_1 number of filter 112 because:
    prev_layer + growth_rate*number_of_conv_layers_in_this_block = 48 + 4*16 = 112
    block 2 - 192: 112 + 5*16 = 192 etc
    '''

    inputs = Input(shape=input_shape)

    first_convolution = Conv2D(48, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)

    block1_down = dense_layer(4, 112, first_convolution)
    block1_down_td = transition_down(112, block1_down)

    block2_down = dense_layer(5, 192, block1_down_td)
    block2_down_td = transition_down(192, block2_down)

    block3_down = dense_layer(7, 304, block2_down_td)
    block3_down_td = transition_down(304, block3_down)

    block4_down = dense_layer(10, 464, block3_down_td)
    block4_down_td = transition_down(464, block4_down)

    block5_down = dense_layer(12, 656, block4_down_td)
    block5_down_td = transition_down(656, block5_down)

    dense_layer_middle = dense_layer(15, 896, block5_down_td)

    block1_up = transition_up(1088, dense_layer_middle, (1088, 7, 7))
    block1_up_dense = dense_layer(12, 1088, block1_up)

    block2_up = transition_up(816, block1_up_dense, (816, 14, 14))
    block2_up_dense = dense_layer(10, 816, block2_up)

    block3_up = transition_up(578, block2_up_dense, (576, 28, 28))
    block3_up_dense = dense_layer(7, 578, block3_up)

    block4_up = transition_up(384, block3_up_dense, (384, 56, 56))
    block4_up_dense = dense_layer(5, 384, block4_up)

    block5_up = transition_up(256, block4_up_dense, (256, 112, 112))
    block5_up_dense = dense_layer(4, 256, block5_up)

    classify = Conv2D(num_classes, kernel_size=(1, 1), padding='same', kernel_regularizer=l2(0.0001), activation='softmax')(block5_up_dense)

    model = Model(inputs=inputs, outputs=classify)

    model.summary()

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model





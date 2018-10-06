# coding:utf-8
'''
video AI model
'''
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Reshape, BatchNormalization
from keras import layers
from keras.optimizers import Adadelta

def Videoai(input_shape, modelarch="xception", output_width=64, output_channel=16):

    # create the base pre-trained model ## todo this should be list
    if modelarch == "xception":
        base_model1 = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model2 = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model3 = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model1 = MobileNet(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model2 = MobileNet(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model3 = MobileNet(weights="imagenet", include_top=False, input_shape=input_shape)

    # change to unique layer name
    for layer in base_model1.layers:
        layer.name = layer.name + str("_one")
    for layer in base_model2.layers:
        layer.name = layer.name + str("_two")
    for layer in base_model3.layers:
        layer.name = layer.name + str("_three")

    frame1_out = base_model1.output
    frame2_out = base_model2.output
    frame3_out = base_model3.output

    frame1_out_rs = Reshape((output_width, output_width, -1))(frame1_out)
    frame2_out_rs = Reshape((output_width, output_width, -1))(frame2_out)
    frame3_out_rs = Reshape((output_width, output_width, -1))(frame3_out)

    x_concat = layers.concatenate([frame1_out_rs, frame2_out_rs, frame3_out_rs])
    x_concat = Conv2D(output_channel, (2, 2), padding='same', name="concat_conv1")(x_concat)
    x_concat = BatchNormalization()(x_concat)
    x_concat = LeakyReLU(alpha=0.2)(x_concat)
    x_concat = Conv2D(output_channel, (2, 2), padding='same', name="concat_conv2")(x_concat)
    x_concat = LeakyReLU(alpha=0.2)(x_concat)
    x_concat = Conv2D(output_channel, (2, 2), padding='same', name="concat_conv3")(x_concat)
    x_concat = BatchNormalization()(x_concat)
    x_concat = LeakyReLU(alpha=0.2)(x_concat)

    model = Model(inputs=[base_model1.input, base_model2.input, base_model3.input],
                  outputs=x_concat)

    optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy']) 

    return model

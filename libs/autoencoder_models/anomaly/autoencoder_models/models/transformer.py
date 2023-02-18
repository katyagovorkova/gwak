import keras
from keras import layers
import numpy as np
import os


'''
models built by:
Zhixing Ethan Jiang
'''
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    # x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = layers.Add()([x, inputs])

    # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res) # just discovered layer norm is not supported, only batch norm supported
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Add()([x, res])
    return x

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units=[10],
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

#     x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(2)(x)
    return keras.Model(inputs, outputs)


def transformer(input_shape:tuple, bottleneck:int):
    #note bottleneck, but just some parameter that can be modified between models
    model = build_model(
    input_shape,
    head_size=20,
    num_heads=4,
    ff_dim=6,
    num_transformer_blocks=3,
    mlp_units=[],
    mlp_dropout=0.4,
    dropout=0.25,
)

    return model, None, None

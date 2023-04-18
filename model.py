from keras.models import Model
from keras.layers import (
    Dense,
    Embedding,
    Conv1D,
    multiply,
    GlobalMaxPool1D,
    Input,
    Activation,
)


def Malconv(max_size=2000000, kernel_size=500, stride_size=500, vocab_size=256):
    inp = Input((max_size,))
    emb = Embedding(vocab_size, 8)(inp)

    conv1 = Conv1D(
        kernel_size=kernel_size, filters=128, strides=stride_size, padding="same"
    )(emb)
    conv2 = Conv1D(
        kernel_size=kernel_size, filters=128, strides=stride_size, padding="same"
    )(emb)
    a = Activation("sigmoid", name="sigmoid")(conv2)

    mul = multiply([conv1, a])
    a = Activation("relu", name="relu")(mul)
    p = GlobalMaxPool1D()(a)
    d = Dense(64)(p)
    out = Dense(1, activation="sigmoid")(d)

    return Model(inp, out)

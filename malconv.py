import tensorflow as tf
import numpy as np
from model import Malconv
import yaml

deploy_model = None
deploy_config = None


def init():
    global deploy_model
    global deploy_config
    # Load deployment setting
    with open("config.yaml", "r") as f:
        deploy_config = yaml.safe_load(f)

    # init model to load pre-trained model's weights only
    deploy_model = Malconv(
        max_size=deploy_config["model"]["max_size"],
        kernel_size=deploy_config["model"]["kernel_size"],
        stride_size=deploy_config["model"]["stride_size"],
    )

    # Load pre-trained model's weights
    deploy_model.load_weights("pre-trained_model.h5")


def padding(byte_sequence, max_size):
    """padding (max_size - len(byte_sequence)) null byte if len(byte_sequence) < max_size
    or truncate byte_sequence if len(byte_sequence) > max_size
    Argument:
        `byte_sequence`: a bytes string
        `max_size`: max_size
    Return:
        a padding sequence with length of max_size
    """
    if len(byte_sequence) > max_size:
        return byte_sequence[:max_size]
    if len(byte_sequence) < max_size:
        return byte_sequence.ljust(max_size, b"\x00")
    return byte_sequence


def get_prediction(sample: bytes):
    """Return a predicted score in [0, 1] for a sample
    Argument:
        `sample`: a sample is presented in bytes string
    Return:
        a predicted score in [0, 1]
    """
    global deploy_model
    global deploy_config

    # init model if it not exist yet
    if deploy_model is None:
        print(
            "Malconv model is not yet initialized. Please wait for model initialization."
        )
        init()

    # convert bytes string sample to numpy sample with shape: (1, max_size)
    np_sample = np.frombuffer(
        padding(sample, deploy_config["model"]["max_size"]), dtype=np.uint8
    ).reshape(1, -1)

    # get prediction in shape: (1,1)
    prediction = deploy_model.predict(np_sample, verbose=0)

    # return prediction values only
    return prediction[0][0]


if __name__ == "__main__":
    # Load a example instance
    s = open("samples/malicious.exe", "rb").read()
    # print prediction
    print(f"Example prediction for a malicious sample: {get_prediction(s)}")

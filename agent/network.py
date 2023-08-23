from dataclasses import dataclass
from typing import Iterable, Literal

import tensorflow as tf  # type: ignore
import tf_agents  # type: ignore


@dataclass
class DropoutDefinition:
    rate: float
    seed: int


@dataclass
class LayerDefinition:
    activation: Literal["relu", "linear"]
    size: int
    seed: int
    dropout: DropoutDefinition | None


def build_dense_layer(definition: LayerDefinition) -> Iterable[tf.keras.layers.Layer]:
    """Factory method to create a dense neural network layer.

    Builds a dense layer with and an directly attached dropout layer (if specified) using
    the given `LayerDefinition`. Layers use `HeNormal` initialization for `ReLU` activations
    and `GlorotNormal` for `linear` activations (i.e. dense layers and output layer).

    :param definition: A `LayerDefinition` describing the layer to create.
    :return: A `tensorflow.keras` neural network layer(s).
    """
    if definition.activation == "relu":
        initializer = tf.keras.initializers.HeNormal(definition.seed)

    if definition.activation == "linear":
        initializer = tf.keras.initializers.GlorotNormal(definition.seed)

    layers = [
        tf.keras.layers.Dense(
            definition.size,
            activation=definition.activation,
            kernel_initializer=initializer,  # type: ignore
        )
    ]

    if definition.dropout:
        layers.extend(
            [
                tf.keras.layers.Dropout(rate=definition.dropout.rate, seed=definition.dropout.seed),
            ]
        )

    return layers


def build_network(observation_size: int, action_size: int) -> tf_agents.networks.Network:
    """Factory method to create the agents network used in this project.

    Dummy values for now.

    :param observation_size: Size of the model input.
    :param action_size: Size if the model output.
    :return: The specified sequential model.
    """
    return tf_agents.networks.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(observation_size,)),
            *build_dense_layer(LayerDefinition("relu", size=768, seed=0, dropout=DropoutDefinition(0.2, seed=0))),
            *build_dense_layer(LayerDefinition("relu", size=512, seed=0, dropout=DropoutDefinition(0.2, seed=0))),
            *build_dense_layer(LayerDefinition("relu", size=256, seed=1, dropout=DropoutDefinition(0.2, seed=1))),
            *build_dense_layer(LayerDefinition("linear", size=action_size, seed=3, dropout=None)),
        ]
    )

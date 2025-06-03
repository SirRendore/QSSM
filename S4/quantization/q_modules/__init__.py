from S4.models.S4 import S4Block
from .qLinear import QLinear
from .qGELU import QGELU
from .qS4D_recurrent import S4BlockRecurrent
from .qS4D_recurrent_undiscretized import S4BlockRecurrentUndiscretized
from .qS4D_recurrent_polar import S4BlockRecurrentPolar
from .qS4D_convolutional import S4BlockConvolutional 
from torch import nn

class qModules():
    S4BlockRecurrent = {
        "class": S4BlockRecurrent,
        "allowed_targets": {
            "S4Block": S4Block
        }
    }

    S4BlockRecurrentUndiscretized = {
        "class": S4BlockRecurrentUndiscretized,
        "allowed_targets": {
            "S4Block": S4Block
        }
    }

    S4BlockRecurrentPolar = {
        "class": S4BlockRecurrentPolar,
        "allowed_targets": {
            "S4Block": S4Block
        }
    }
    S4BlockConvolutional = {
        "class": S4BlockConvolutional,
        "allowed_targets": {
            "S4Block": S4Block
        }
    }

    QLinear = {
        "class": QLinear,
        "allowed_targets": {
            "Linear": nn.Linear
        }
    }

    QGELU = {
        "class": QGELU,
        "allowed_targets": {
            "GELU": nn.GELU
        }
    }
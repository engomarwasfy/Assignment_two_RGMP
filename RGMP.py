import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models
from Encoder import Encoder
from Decoder import Decoder

class RGMP(nn.Module):
    def __init__(self):
        super(RGMP, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

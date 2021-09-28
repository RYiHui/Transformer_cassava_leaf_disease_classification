"""
    Cassava
"""

import torch
import torchvision.transforms as transforms
import numpy  as np
import pandas as pd
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn import model_selection,metrics
import CassavaDataset
import Global_Variable as gl
import run

if __name__ == '__main__':
    torch.set_default_tensor_type("torch.FloatTensor")
    a = run.run()




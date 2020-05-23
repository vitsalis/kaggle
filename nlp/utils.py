import os
import random
import torch as th
import numpy as np

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    th.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed_value)
        th.cuda.manual_seed_all(seed_value)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = True

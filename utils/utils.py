import os
from datetime import datetime
import random

import numpy as np
import torch

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True was used for the paper experiments; results are therefore not bit-exact across runs
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_date():
    now = datetime.now()
    date = now.strftime('%Y%m%d_%H%M%S')
    return date

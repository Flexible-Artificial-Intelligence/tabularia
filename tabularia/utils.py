import numpy as np
from typing import Optional
import random


def get_random_number(min_value:int=0, max_value:int=50) -> int:
    """
    Returns random value from [`min_value`, `max_value`] range.
    """
    
    return random.randint(min_value, max_value)


def seed_everything(seed:Optional[int]=None) -> int:
    """
    Sets seed for `numpy` and `random` libraries to have opportunity to reproduce results.
    """
    if seed is None:
        seed = get_random_number()
        
    random.seed(seed)
    np.random.seed(seed)
    
    return seed
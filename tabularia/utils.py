import pandas as pd
import numpy as np
from typing import Optional
import random
import os

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold


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


def read_data_frame(path, library=pd, *args, **kwargs):
    file_extension = os.path.splitext(path)[0][1:]
    read_function_name = f"read_{file_extension}"
    read_function = getattr(library, read_function_name)
    data_frame = read_function(path, *args, **kwargs)
    
    return data_frame


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_folds(data_frame, 
                 targets, 
                 groups=None, 
                 stratify=False, 
                 return_data_frame=True,  
                 fold_column="fold", 
                 folds=5,
                 **kwargs):

    if groups is None:
        if stratify:
            cv_strategy = StratifiedKFold(n_splits=folds, **kwargs)
        else:
            cv_strategy = KFold(n_splits=folds, **kwargs)
            
        folds = cv_strategy.split(X=data_frame, y=targets)
    else:
        if stratify:
            cv_strategy = StratifiedGroupKFold(n_splits=folds, **kwargs)
        else:
            cv_strategy = GroupKFold(n_splits=folds, **kwargs)
            
        folds = cv_strategy.split(X=data_frame, y=targets, groups=groups)
    
    if return_data_frame:
        for fold, (train_indexes, validation_indexes) in enumerate(folds):
            data_frame.loc[validation_indexes, fold_column] = int(fold+1)

        return data_frame
    
    return folds
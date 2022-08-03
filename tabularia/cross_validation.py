import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class CrossValidation:
    def __init__(self, 
                 model, 
                 splitter=KFold,
                 model_parameters={}, 
                 scorer=None, 
                 verbose=True):
        
        self.model = model
        self.model_parameters = model_parameters
        self.splitter = splitter
        self.scorer = scorer
        self.verbose = verbose
        
        self.__num_folds = self.splitter.n_splits
    
        
    def get_split(self, dataset, targets, groups=None):
        if groups is not None:
            folds = self.splitter.split(X=dataset, y=targets, groups=groups)
        else:
            folds = self.splitter.split(X=dataset, y=targets)
            
        return folds

        
    def get_fold_data(self, dataset, targets, train_indexes, validation_indexes):
        if isinstance(dataset, pd.DataFrame):
            train_data = dataset.iloc[train_indexes]
            validation_data = dataset.iloc[validation_indexes]
        else:
            train_data = dataset[train_indexes]
            validation_data = dataset[validation_indexes]
        
        train_targets = targets[train_indexes]
        validation_targets = targets[validation_indexes]
        
        return (train_data, train_targets), (validation_data, validation_targets)
        
    def initialize_model(self, model=None, model_parameters=None):
        if model is None:
            model = self.model
        
        if model_parameters is None:
            model_parameters = self.model_parameters
            
        model = model(**model_parameters)
        return model
    
    
    def score(self, targets, predictions):
        if self.scorer is not None:
            score = self.scorer(targets, predictions)
            return score
        
        return np.nan
    
    
    def predict(self, model, data):
        return model.predict(data)
    
    def fit(self, 
            model, 
            train_data, 
            train_targets, 
            validation_data, 
            validation_targets):
        
        model.fit(train_data, train_targets)
        
    def save_model(self, model, fold):
        return None
        
    def __call__(self, dataset, targets, groups=None, test_dataset=None):
        scores, oof_predictions, test_predictions = [], [], []
        
        folds = self.get_split(dataset, targets)
        for fold, (train_indexes, validation_indexes) in enumerate(folds):
            (train_data, train_targets), (validation_data, validation_targets) = self.get_fold_data(dataset=dataset, 
                                                                                                    targets=targets,
                                                                                                    train_indexes=train_indexes, 
                                                                                                    validation_indexes=validation_indexes)
            
            model = self.initialize_model(model=self.model, model_parameters=self.model_parameters)
            self.fit(model=model, 
                     train_data=train_data, 
                     train_targets=train_targets, 
                     validation_data=validation_data, 
                     validation_targets=validation_targets)
            
            self.save_model(model=model, fold=fold)
            
            validation_predictions = self.predict(model=model, data=validation_data)
            validation_score = self.score(targets=validation_targets, predictions=validation_predictions)
            
            oof_predictions.append(validation_predictions)
            scores.append(validation_score)
            
            if self.verbose:
                print(f"Fold {fold+1}/{self.__num_folds}: {validation_score}", end="\n"*2)
            
            if test_dataset is not None:
                fold_test_predictions = self.predict(model=model, data=test_dataset)
                test_predictions.append(fold_test_predictions)
        
            
        oof_predictions = np.array(oof_predictions)
        scores = np.array(scores)
        test_predictions = np.array(test_predictions)
            
        if self.verbose:
            cv_mean = np.mean(scores)
            cv_std = np.std(scores)

            print(f"CV {cv_mean} +- {cv_std}: {scores}")


        if test_dataset is not None:
            return scores, oof_predictions, test_predictions

        return scores, oof_predictions
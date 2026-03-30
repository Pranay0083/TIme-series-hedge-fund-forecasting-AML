import numpy as np

class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Split.
    Ensures strict temporal splitting with a specified gap to prevent look-ahead bias,
    specifically tailored for financial target horizons.
    """
    def __init__(self, n_splits=5, gap=31, max_train_size=None):
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        
    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.
        groups must be the temporal index (ts_index).
        """
        if groups is None:
            raise ValueError("The 'groups' parameter must be provided.")
            
        group_unique = np.sort(np.unique(groups))
        n_groups = len(group_unique)
        fold_size = n_groups // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end_idx = fold_size * (i + 1)
            
            if self.max_train_size and train_end_idx > self.max_train_size:
                train_start_idx = train_end_idx - self.max_train_size
            else:
                train_start_idx = 0
                
            test_start_idx = train_end_idx + self.gap
            test_end_idx = test_start_idx + fold_size
            
            if test_start_idx >= n_groups:
                continue
            if test_end_idx > n_groups:
                test_end_idx = n_groups
                
            train_groups = group_unique[train_start_idx:train_end_idx]
            test_groups = group_unique[test_start_idx:test_end_idx]
            
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            
            yield train_idx, test_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

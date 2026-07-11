import os
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data


def stratified_split(df, features=None, target=None,
                     test_size=None, random_state=None):
    """Single source of truth for the train/test holdout split.

    Stratified on the target so the ~0.17% fraud rate is preserved in both
    splits. This is deliberately SEPARATE from the StratifiedKFold folds that
    Hyperopt uses inside its objective for cross-validation: this holdout is the
    final, untouched evaluation set; the CV folds live entirely within X_train.

    Returns X_train, X_test, y_train, y_test.
    """
    features = config.FEATURES if features is None else features
    target = config.TARGET if target is None else target
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    X = df[features]
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

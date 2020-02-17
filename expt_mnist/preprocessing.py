import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.utility import log_shapes, load_processed_data, to_categorical


def get_data(expt, test_size):

    X, y_ally, y_advr_1, y_advr_2 = load_processed_data(
        expt,
        'processed_data_X_y_ally_y_advr_y_advr_2.pkl'
    )
    log_shapes(
        [X, y_ally, y_advr_1, y_advr_2],
        locals(),
        'Dataset loaded'
    )

    X_train, X_valid, \
        y_ally_train, y_ally_valid, \
        y_advr_1_train, y_advr_1_valid, \
        y_advr_2_train, y_advr_2_valid = train_test_split(
            X,
            y_ally,
            y_advr_1,
            y_advr_2,
            test_size=test_size,
            stratify=pd.DataFrame(np.concatenate(
                (
                    y_ally.reshape(-1, 1),
                    y_advr_1.reshape(-1, 1),
                    y_advr_2.reshape(-1, 1),
                ), axis=1)
            )
        )

    scaler = MinMaxScaler()
    scaler.fit(X_train.astype(np.float64))
    X_normalized_train = scaler.transform(X_train.astype(np.float64))
    X_normalized_valid = scaler.transform(X_valid.astype(np.float64))

    y_ally_train = y_ally_train.reshape(-1, 1)
    y_ally_valid = y_ally_valid.reshape(-1, 1)
    y_advr_1_train = to_categorical(y_advr_1_train)
    y_advr_1_valid = to_categorical(y_advr_1_valid)
    y_advr_2_train = y_advr_2_train.reshape(-1, 1)
    y_advr_2_valid = y_advr_2_valid.reshape(-1, 1)

    log_shapes(
        [
            X_normalized_train, X_normalized_valid,
            y_ally_train, y_ally_valid,
            y_advr_1_train, y_advr_1_valid,
            y_advr_2_train, y_advr_2_valid,
        ],
        locals(),
        'Data size after train test split'
    )

    return (
        X_normalized_train, X_normalized_valid,
        y_ally_train, y_ally_valid,
        y_advr_1_train, y_advr_1_valid,
        y_advr_2_train, y_advr_2_valid,
    )

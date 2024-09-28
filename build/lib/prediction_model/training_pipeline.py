import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.processing import preprocessing as pp
import prediction_model.pipeline as pipe
from pathlib import Path
import sys
import os


# # Adding the below path to avoid module not found error
# PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
# sys.path.append(str(PACKAGE_ROOT))


def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({'N':0, 'Y': 1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    #print(pipe.classification_pipeline)
    save_pipeline(pipe.classification_pipeline)


if __name__=='__main__':
    perform_training()




from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


classification_pipeline = Pipeline(
    steps = [
        ('DomainProcessing', pp.DomainProcessing(variables_to_modify=config.FEATURE_TO_MODIFY, variables_to_add=config.FEATURE_TO_ADD)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoding', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTranform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)
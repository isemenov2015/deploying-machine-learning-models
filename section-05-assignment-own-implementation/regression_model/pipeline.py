from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regression_model.config.core import config
from regression_model.processing import features as pp

titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars,
            ),
        ),
        # ==== VARIABLE TRANSFORMATION =====
        (
            "1st letter transformation",
            pp.ExtractLetterTransformer(variables=config.model_config.cabin_vars)
        ),

        # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        # encode categorical variables using one-hot encoder
        (
            "categorical_encoder",
            OneHotEncoder(
                variables=config.model_config.categorical_vars,
                ignore_format=True
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "Logistic regression",
            LogisticRegression(
                C=config.model_config.C_logit,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)

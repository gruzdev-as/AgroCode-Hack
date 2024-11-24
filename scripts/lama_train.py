import os
import random
import pandas as pd
import numpy as np
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from lightautoml.report.report_deco import ReportDeco


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def log_step(message, char='-', width=80):
    separator = char * width
    print(f'\n{separator}\n{message}\n{separator}')


def train_lama(
    train_path=r'data/train_file.csv',
    n_threads=4,
    n_folds=7,
    random_state=52,
    timeout=36000,
    target_name='target'
):
    log_step('Initializing training process')

    N_THREADS = n_threads
    N_FOLDS = n_folds
    RANDOM_STATE = random_state
    TIMEOUT = timeout
    TARGET_NAME = target_name

    log_step(f'Setting random seed: {RANDOM_STATE}')
    seed_everything(RANDOM_STATE)

    log_step(f'Loading training data from: {train_path}')
    train_with_weather = pd.read_csv(train_path)

    X = train_with_weather.drop('target', axis=1)
    y = train_with_weather.target

    embedding_features = [
        '1', '10', '11', '12', '13', '14', '15', '16', '17',
        '18', '19', '2', '20', '3', '4', '5', '6', '7', '8', '9'
    ]

    log_step('Processing embedding features')
    for feature in embedding_features:
        X[feature] = X[feature].apply(lambda x: x[1:-1].split())

    for col in embedding_features:
        expanded_cols = X[col].apply(pd.Series)
        expanded_cols.columns = [f'{col}_feature_{i+1}' for i in range(expanded_cols.shape[1])]
        X = pd.concat([X, expanded_cols], axis=1)

    X = X.drop(columns=embedding_features)

    train_df = pd.concat([X, y], axis=1)

    log_step('Initializing TabularAutoML')
    automl = TabularAutoML(
        task=Task('reg', loss='mse', metric='r2'),
        reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE, 'advanced_roles': False},
        tuning_params={'max_tuning_time': (TIMEOUT // 10) * 3, 'max_tuning_iter': 200, 'fit_on_holdout': True},
        timeout=TIMEOUT,
        cpu_limit=N_THREADS
    )

    log_step('Starting training')
    oof_pred = automl.fit_predict(train_df, roles={'target': TARGET_NAME}, verbose=1)

    log_step('Model training summary')
    print(automl.create_model_str_desc())

    log_step('Calculating metrics')
    r2 = r2_score(train_df[TARGET_NAME].values, oof_pred.data.flatten())
    mape = mean_absolute_percentage_error(train_df[TARGET_NAME].values, oof_pred.data.flatten())
    mae = mean_absolute_error(train_df[TARGET_NAME].values, oof_pred.data.flatten())
    print(
        f'TRAIN out-of-fold metrics:\n'
        f' - R2 Score: {r2:.4f}\n'
        f' - MAPE Score: {mape:.4f}\n'
        f' - MAE Score: {mae:.4f}\n'
    )

    log_step('Generating model report')
    RD = ReportDeco(output_path='tabularAutoML_model_report')
    automl_rd = RD(automl)

    accurate_fi = automl_rd.model.get_feature_scores('fast')
    print('\nTop features by importance:')
    print(accurate_fi.head(10))
    accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize=(50, 20), grid=True)

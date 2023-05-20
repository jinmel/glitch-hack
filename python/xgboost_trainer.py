"""Train fraud detection model and export it as ONNX format."""

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import onnxruntime as rt


FLAGS = flags.FLAGS
flags.DEFINE_string('output', None, 'Output path of exported onnx model.')


def load_data(csv_filename):
    """Load eth fraud dataset from csv."""

    df = pd.read_csv(csv_filename, index_col=0)
    df = df.iloc[:, 2:]
    # Drop object columns
    categories = df.select_dtypes('O').columns.astype('category')
    df.drop(df[categories], axis=1, inplace=True)
    df.fillna(df.median(), inplace=True)

    # Drop columns with variance == 0
    df.drop(df.var()[df.var() == 0].index, axis=1, inplace=True)

    # Drop highly correlated features.
    drop = ['total transactions (including tnx to create contract',
        'total ether sent contracts',
        'max val sent to contract',
        ' ERC20 avg val rec',
        ' ERC20 avg val rec',
        ' ERC20 max val rec',
        ' ERC20 min val rec',
        ' ERC20 uniq rec contract addr',
        'max val sent',
        ' ERC20 avg val sent',
        ' ERC20 min val sent',
        ' ERC20 max val sent',
        ' Total ERC20 tnxs',
        'avg value sent to contract',
        'Unique Sent To Addresses',
        'Unique Received From Addresses',
        'total ether received',
        ' ERC20 uniq sent token name',
        'min value received',
        'min val sent',
        ' ERC20 uniq rec addr']
    # Drop mostly 0
    drop += ['min value sent to contract', ' ERC20 uniq sent addr.1']
    df.drop(drop, axis=1, inplace=True)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    return X, y


def train_model(X, y):
    X, y = load_data('./transaction_dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 123)

    sc = StandardScaler()
    sc_train = sc.fit_transform(X_train)
    sc_test = sc.transform(X_test)

    oversample = SMOTE()
    x_tr_resample, y_tr_resample = oversample.fit_resample(sc_train, y_train)

    # Train XGBClassifier.
    xgb_c = xgb.XGBClassifier(random_state=42, n_estimators=3)
    xgb_c.fit(x_tr_resample, y_tr_resample)
    preds_xgb = xgb_c.predict(sc_test)
    print(classification_report(y_test, preds_xgb))
    print(confusion_matrix(y_test, preds_xgb))


def main(_):
    X, y = load_data('./transaction_dataset.csv')
    train_model(X, y)

    oversample = SMOTE()
    X_resample, y_resample = oversample.fit_resample(X, y)

    # Define and train XGBClassifier pipeline
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('xgb', xgb.XGBClassifier(n_estimators=3))])
    pipe.fit(X_resample, y_resample)

    update_registered_converter(
        xgb.XGBClassifier, 'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
    model_onnx = convert_sklearn(pipe, 'pipeline_xgboost',
                    [('input', FloatTensorType([None, X.shape[1]]))],
                    target_opset={'': 12, 'ai.onnx.ml': 2})

    print(model_onnx)

    with open(FLAGS.output, 'wb') as f:
        f.write(model_onnx.SerializeToString())
    logging.info('Done writing to %s.', FLAGS.output)
    logging.info('Testing onnx inference...')
    sess = rt.InferenceSession(FLAGS.output)
    input_X = X.to_numpy()
    pred_onnx = sess.run(None, {"input": input_X[:30].astype(np.float32)})
    print('predict', pred_onnx)


if __name__ == '__main__':
    app.run(main)

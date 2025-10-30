# IMPORTS --------------------------

import csv
import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.getcwd()))

from utils import eda_helpers, all_functions, cv, sampling_methods
import implementations

from utils.eda_helpers import clean_data, read_csv_numeric
from utils.all_functions import sigmoid
from implementations import (
    logistic_regression,
    reg_logistic_regression,
    weighted_logistic_regression,
)
from utils.cv import train_logreg, predict
from utils.helpers import load_csv_data, create_csv_submission

file_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FOLDER = os.path.join(file_path, 'dataset')

# LOADING DATA ----------------------------------------------------------

# load data
x_indices, x_data = read_csv_numeric(os.path.join(DATA_FOLDER, "x_train.csv"))
y_indices, y_data = read_csv_numeric(os.path.join(DATA_FOLDER, "y_train.csv"))
print(f"Loaded train data")
x_indices_test, x_test = read_csv_numeric(os.path.join(DATA_FOLDER, "x_test.csv"))
print(f"Loaded test data")

# clean train data
del x_indices["Id"]
x_data = clean_data(x_data, x_indices)
print(f"Cleaned train data")


# clean test
if "Id" in x_indices_test:
    del x_indices_test["Id"]
x_test = clean_data(x_test, x_indices_test)
print(f"Cleaned test data")


# final feature set
final_feature_set = ['_CASTHM1', 'FLUSHOT6', 'METVL11_', '_PASTRNG', 'EDUCA', 'CHCCOPD1', 'DRADVISE', 
                     'CHCKIDNY', 'ARTHSOCL', 'MENTHLTH', 'SMOKE100', 'DOCTDIAB', 'DIABAGE2', 'VETERAN3', 
                     'CHKHEMO3', 'SEX', '_DRNKWEK', 'JOINPAIN', 'MARITAL', 'EXERANY2', '_PACAT1', 'INSULIN', 
                     'DIFFALON', 'FRUIT1', 'QLACTLM2', 'BLIND', 'PADUR2_', 'PAFREQ2_', 'LMTJOIN3', 'CHOLCHK', 
                     'POORHLTH', 'DIABETE3', '_FRUTSUM', 'PERSDOC2', 'EXRACT21', 'FEETCHK', '_VEGESUM', 'PNEUVAC3', 
                     'BPHIGH4', 'GENHLTH', 'EXRACT11', '_BMI5', 'DECIDE', '_ASTHMS1', '_MRACE1', '_LMTSCL1', 'ORNGDAY_', 
                     'MSCODE', 'PHYSHLTH', 'TOLDHI2', 'PADUR1_', 'DIFFWALK', 'BEANDAY_', 'CHCSCNCR', 'PAFREQ1_', '_AIDTST3', 
                     'FLSHTMY2', 'PA1VIGM_', 'METVL21_', 'SXORIENT', 'ARTHDIS2', 'PA1MIN_', 'MAXVO2_', '_RACE_G1', 'LONGWTCH', 
                     'SCNTWRK1', 'FEETCHK2', 'EMPLOY1', 'BLOODCHO', 'USENOW3', 'DRNKANY5', '_PAREC1', 'DIFFDRES', 'USEEQUIP', 
                     'ADDEPEV2', '_AGE80', 'HTM4', 'INCOME2', 'CVDSTRK3', 'CHCOCNCR', 'MEDCOST', 'PREDIAB1', 'BPMEDS', 'HAVARTH3', 
                     'CHECKUP1', 'TRNSGNDR', 'CHILDREN']

# PREPROCESS TRAIN DATA--------------

# keep only the columns of interest
x_data_interested = x_data[:, [x_indices[col] for col in final_feature_set]]
x_indices_interested = {col: i for i, col in enumerate(final_feature_set)}

# drop columns that are all Na
valid_cols = ~np.isnan(x_data_interested).all(axis=0)
x_data_interested = x_data_interested[:, valid_cols]
interesting_columns = [col for col, keep in zip(final_feature_set, valid_cols.tolist()) if keep] # type: ignore
x_indices_interested = {col: i for i, col in enumerate(interesting_columns)}

# impute remaining NaNs with column mean on train
col_mean = np.nanmean(x_data_interested, axis=0)
inds = np.where(np.isnan(x_data_interested))
x_data_interested[inds] = np.take(col_mean, np.array(inds[1], dtype=int))

# compute std
col_std = np.nanstd(x_data_interested, axis=0)
col_std[col_std == 0] = 1

# standardize once
x_data_standardized = (x_data_interested - col_mean) / col_std

# save train stats for standardizing test later
train_stats = {"mean": col_mean, "std": col_std, "columns": interesting_columns}

# prepare y
y = y_data[:,1].copy()
y[np.isnan(y)] = np.nanmean(y)
y = (y + 1) / 2  # convert -1,1 → 0,1
y = y.flatten()

# checks
print("Unique labels:", np.unique(y))
print("Any NaN in features?", np.any(np.isnan(x_data_standardized)))
print("Min std across features:", np.min(np.std(x_data_standardized, axis=0)))
print("Label distribution (mean):", np.mean(y))
print("Shape after cleaning:", x_data_standardized.shape)

# TEST DATA PREPROCESS --------------
# keep only same features as in training 
x_test_interested = x_test[:, [x_indices_test[col] for col in interesting_columns]]

# impute missing values using training column means 
inds_test = np.where(np.isnan(x_test_interested))
x_test_interested[inds_test] = np.take(col_mean, inds_test[1])

# standardize using training mean/std 
x_test_standardized = (x_test_interested - col_mean) / col_std

# TRAIN ---------------------------------------

lambda_ = 0.01
gamma = 0.4
max_iters = 500
threshold = 0.80

print("\n=== Training Weighted Logistic Regression ===")
w, final_loss = train_logreg(
    y=y,
    tx=x_data_standardized,
    kind="w_logistic",
    gamma=gamma,
    lambda_=lambda_,
    max_iters=max_iters,
    tol=None,
    verbose=True
)
print(f"Final training loss: {final_loss:.6f}")


# predict on test set
print("\n=== Generating Predictions ===")
y_pred_test = predict(x_test_standardized, w, threshold=threshold)

# convert 0,1 to -1,1 for AI Crowd submission
y_pred_test = np.where(y_pred_test == 1, 1, -1)


# load test IDs and create submission file
test_ids = np.genfromtxt(
    os.path.join(DATA_FOLDER, "x_test.csv"),
    delimiter=",",
    skip_header=1,
    usecols=[0],
    dtype=int
)

submission_path = os.path.join(file_path, f"submission_weighed_log_reg_gamma{gamma}_lambda{lambda_}_thr{threshold}.csv")
create_csv_submission(test_ids, y_pred_test, submission_path)
print(f"\nSubmission file successfully created: {submission_path}")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Any


def read_files(
    train_folder_path: str,
    pred_folder_path: str,
    sep: str = '|',
    encoding: str = 'big5',
    dropna: bool = False
) -> (pd.DataFrame, pd.DataFrame):
    """
    Reads and concatenates CSV files from the specified train and prediction folders.

    :param train_folder_path: Path to the folder containing training files.
    :param pred_folder_path: Path to the folder containing prediction files.
    :param sep: Delimiter used in CSV files.
    :param encoding: Encoding of CSV files.
    :param dropna: Whether to drop rows with NA after reading.
    :return: Two DataFrames: reference (training) and current (prediction).
    """
    train_file_list = [
        os.path.join(train_folder_path, f) for f in os.listdir(train_folder_path)
        if f.lower().endswith('.csv')
    ]
    pred_file_list = [
        os.path.join(pred_folder_path, f) for f in os.listdir(pred_folder_path)
        if f.lower().endswith('.csv')
    ]

    print("[INFO] Training files:", train_file_list)
    print("[INFO] Prediction files:", pred_file_list)

    reference = pd.concat(
        [pd.read_csv(file, sep=sep, encoding=encoding) for file in train_file_list],
        ignore_index=True
    )
    current = pd.concat(
        [pd.read_csv(file, sep=sep, encoding=encoding) for file in pred_file_list],
        ignore_index=True
    )

    if dropna:
        reference.dropna(inplace=True)
        current.dropna(inplace=True)

    return reference, current


def clean_data(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    val_size: float = 0.2,
    drop_columns: Optional[List[str]] = None,
    impute_zero_cols: Optional[List[str]] = None,
    rename_prefix: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Cleans and preprocesses data, then splits into train/validation sets.

    :param reference: DataFrame for training/validation.
    :param current: DataFrame for prediction.
    :param val_size: Validation set size fraction (0 < val_size < 1).
    :param drop_columns: List of column names to drop.
    :param impute_zero_cols: List of columns where NaN should be replaced with 0.
    :param rename_prefix: If provided, columns will be renamed with an index-based prefix (e.g. "01_colname").
    :param random_state: Random state for train_test_split.
    :return: A dict with keys {'ref': DataFrame, 'val': DataFrame, 'cur': DataFrame}.
    """
    # 1. Imputation
    if impute_zero_cols:
        for col in impute_zero_cols:
            reference.loc[reference[col].isnull(), col] = 0
            current.loc[current[col].isnull(), col] = 0

    # 2. Train/validation split
    if 0 < val_size < 1:
        reference, validation = train_test_split(
            reference,
            test_size=val_size,
            random_state=random_state
        )
    else:
        # If val_size = 0 or invalid, assume entire reference as "train" and empty validation
        validation = pd.DataFrame([])

    # 3. Drop unwanted columns
    if drop_columns:
        reference.drop(columns=drop_columns, errors='ignore', inplace=True)
        validation.drop(columns=drop_columns, errors='ignore', inplace=True)
        current.drop(columns=drop_columns, errors='ignore', inplace=True)

    # 4. Rename columns with an enumerated prefix if needed
    if rename_prefix:
        # Example rename: "01_colA", "02_colB"...
        new_reference_cols = {
            col: f"{str(i+1).zfill(2)}_{col}" for i, col in enumerate(reference.columns)
        }
        reference.rename(columns=new_reference_cols, inplace=True)
        validation.rename(columns=new_reference_cols, inplace=True)
        current.rename(columns=new_reference_cols, inplace=True)

    # 5. Drop any remaining NaN rows if that’s desired
    reference.dropna(inplace=True)
    validation.dropna(inplace=True)
    current.dropna(inplace=True)

    return {'ref': reference, 'val': validation, 'cur': current}

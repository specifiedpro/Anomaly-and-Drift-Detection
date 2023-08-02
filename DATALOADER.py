import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def read_files(train_folder_path, pred_folder_path, output_path): # Read raw data and concatenate the reference data.
    # 訓練資料夾路徑、預測資料夾路徑
    train_folder, predict_folder = train_folder_path, pred_folder_path
    # 建立空的列表存储檔案路徑
    ref_file_list, cur_file_list = [], []
    
    # 列出訓練資料夾中的所有檔案
    for filename in os.listdir(train_folder):
        file_path = os.path.join(train_folder, filename)
        ref_file_list.append(file_path)
    # 列出預測資料夾中的所有檔案
    for filename in os.listdir(predict_folder):
        file_path = os.path.join(predict_folder, filename)
        cur_file_list.append(file_path)
        

    print(ref_file_list, '|', cur_file_list)
    reference = pd.concat([pd.read_csv(file, sep='|', encoding='big5') for file in ref_file_list])
    current = pd.concat([pd.read_csv(file, sep='|', encoding='big5') for file in cur_file_list], ignore_index=True)
    

    return reference, current

def clean_data(reference, current, output_path, val_size = 0): # Prepocess data and save the cleaned dataframe.
    # This function return a dictionary {'ref':pd.DataFrame, 'cur':pd.DataFrame, 'val':pd.DataFrame}
    data_dict = {}
    # 1. Data Imputation
    reference.loc[reference['L1MN_L3MN_GAME_SOCIAL_CNT_RT'].isnull(), 'L1MN_L3MN_GAME_SOCIAL_CNT_RT'] = 0
    reference.loc[reference['L1MN_L3MN_GAME_SOCIAL_AMT_RT'].isnull(), 'L1MN_L3MN_GAME_SOCIAL_AMT_RT'] = 0
    current.loc[current['L1MN_L3MN_GAME_SOCIAL_CNT_RT'].isnull(), 'L1MN_L3MN_GAME_SOCIAL_CNT_RT'] = 0
    current.loc[current['L1MN_L3MN_GAME_SOCIAL_AMT_RT'].isnull(), 'L1MN_L3MN_GAME_SOCIAL_AMT_RT'] = 0

    
    reference, validation = train_test_split(reference, test_size=0.2, random_state=42)
    new_columns = {col: f"{i}_{col}" for i, col in enumerate(reference.columns)}
    
    reference = reference.drop(['MOST_USE_BRAND', 'SUBSCR_ID', 'STATIS_MN','L6MN_MP_CNT','L3MN_MP_CNT'], axis=1)
    validation = validation.drop(['MOST_USE_BRAND', 'SUBSCR_ID', 'STATIS_MN','L6MN_MP_CNT','L3MN_MP_CNT'], axis=1)
    current = current.drop(['MOST_USE_BRAND', 'SUBSCR_ID', 'STATIS_MN','L6MN_MP_CNT','L3MN_MP_CNT'], axis=1)
    
    new_columns = {col: f"{str(i+1).zfill(2)}_{col}" for i, col in enumerate(reference.columns)}
    reference = reference.rename(columns=new_columns)
    validation = validation.rename(columns=new_columns)
    current = current.rename(columns=new_columns)

    reference = reference.dropna()
    validation = validation.dropna()
    current = current.dropna()
    data_dict = {'ref':reference, 'val':validation, 'cur':current}
    
    
    return data_dict
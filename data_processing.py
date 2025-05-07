
import numpy as np
import pandas as pd
import re
from sklearn.utils import resample

data = pd.read_csv("C:/Users/RE612/Desktop/112-1/ml/hw1/archive/train.csv")

# --------------- Drop ID ---------------#
data = data.drop('policy_id', axis=1)


# --------------- Column - max_torque ---------------#
data['torque'] = data['max_torque'].apply(
    lambda x: re.findall(r'\d+\.?\d*(?=Nm)', x)[0])
data['rpm'] = data['max_torque'].apply(
    lambda x: re.findall(r'\d+\.?\d*(?=rpm)', x)[0])

# Convert the columns to numeric data type
data['torque'] = pd.to_numeric(data['torque'])
data['rpm'] = pd.to_numeric(data['rpm'])

# Calculate torque to RPM ratio
data['torque to rpm ratio'] = data['torque'] / data['rpm']

# Deleting redundant columns from training set
data.drop('max_torque', axis=1, inplace=True)
data.drop('rpm', axis=1, inplace=True)
data.drop('torque', axis=1, inplace=True)


# --------------- Column - max_power ---------------#
data['power'] = data['max_power'].apply(
    lambda x: re.findall(r'\d+\.?\d*(?=bhp)', x)[0])
data['rpm'] = data['max_power'].apply(lambda x: re.findall(r'\d+', x)[-1])

# Convert the columns to numeric data type
data['power'] = pd.to_numeric(data['power'])
data['rpm'] = pd.to_numeric(data['rpm'])

# Calculate power to RPM ratio
data['power to rpm ratio'] = data['power'] / data['rpm']

data.drop('power', axis=1, inplace=True)
data.drop('rpm', axis=1, inplace=True)
data.drop('max_power', axis=1, inplace=True)


# --------------- Column - is___ ---------------#
data = data.replace({"No": 0, "Yes": 1})


# --------------- Column - all categorical data ---------------#
dataset_cat_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=dataset_cat_cols, drop_first=True)


# --------------- SMOTE --------------- #
print(data.loc[:, "is_claim"].value_counts())


X = data.drop("is_claim", axis=1)
y = data.loc[:, "is_claim"]


def smote_oversampling(X, y, k=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    X_resampled = []
    y_resampled = []

    # 确定两个类别
    classes = np.unique(y)

    # 确定少数类别和多数类别
    if len(X[y == classes[0]]) < len(X[y == classes[1]]):
        minority_class = classes[0]
        majority_class = classes[1]
    else:
        minority_class = classes[1]
        majority_class = classes[0]

    # 获取少数类别样本的索引
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # 计算需要生成的合成样本数量
    num_samples_to_generate = len(majority_indices) - len(minority_indices)

    for _ in range(num_samples_to_generate):
        # 随机选择一个少数类别样本
        minority_index = np.random.choice(minority_indices)

        # 随机选择k个近邻样本
        neighbors = np.random.choice(majority_indices, k, replace=True)

        alpha = np.random.random()

        # 生成新的合成样本
        new_sample = X[minority_index] + alpha * \
            (X[neighbors[0]] - X[minority_index])

        X_resampled.append(new_sample)
        y_resampled.append(minority_class)

    # 合并原始样本和合成样本
    X_resampled = np.vstack((X, np.array(X_resampled)))
    y_resampled = np.concatenate((y, np.array(y_resampled)))

    return X_resampled, y_resampled


X_resampled, y_resampled = smote_oversampling(
    X.values, y.values, random_seed=42)

X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.DataFrame(
    y_resampled, columns=['is_claim'])

data_final = pd.concat([X_resampled_df, y_resampled_df], axis=1)

df = data_final.replace({False: 0, True: 1})
df.reset_index(inplace=True, drop=True)

# print(df)
# print(df.info())
print(df.shape)
print(df.loc[:, "is_claim"].value_counts())

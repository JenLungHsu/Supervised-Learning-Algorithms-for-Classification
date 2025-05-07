
import time
from scikit_posthocs import posthoc_nemenyi
from scipy.stats import friedmanchisquare
import pandas as pd
import numpy as np
import definition_model as model
import data_processing as df
from sklearn.metrics import accuracy_score, classification_report

X = df.df.drop("is_claim", axis=1)
y = df.df.loc[:, "is_claim"]

X['volumn'] = X['height'] * X['width'] * X['length']

newX = X.loc[:, ['volumn', 'height', 'width', 'length',
                 'age_of_car', 'torque to rpm ratio', 'area_cluster_C20', 'population_density',
                 'gross_weight', 'displacement', 'policy_tenure', 'ncap_rating', 'airbags', 'make', 'area_cluster_C15', 'cylinder', 'is_adjustable_steering', 'is_day_night_rear_view_mirror', 'fuel_type_Diesel', 'transmission_type_Manual', 'area_cluster_C8', 'is_tpms', 'is_ecw', 'engine_type_K10C',
                 'is_parking_sensors',
                 'is_power_door_locks', 'area_cluster_C18', 'engine_type_1.5 Turbocharged Revotron', 'is_driver_seat_height_adjustable', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'engine_type_K Series Dual jet', 'is_central_locking', 'is_power_steering', 'engine_type_G12B', 'area_cluster_C15']]

X_train, X_test, y_train, y_test = model.split_data(newX, y)


def custom_k_fold_cross_validation_pd(X, y, model, k_folds=5):
    fold_size = len(X) // k_folds
    accuracies = []

    for i in range(k_folds):
        # 划分训练集和测试集
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        test_X = X.iloc[test_start:test_end]  # 使用iloc来选择DataFrame的行
        test_y = y.iloc[test_start:test_end]

        train_X = pd.concat([X.iloc[:test_start], X.iloc[test_end:]], axis=0)
        train_y = pd.concat([y.iloc[:test_start], y.iloc[test_end:]], axis=0)

        # 在训练集上训练你的机器学习模型
        model.fit(train_X, train_y)

        # 在测试集上进行预测
        y_pred = model.predict(test_X)

        # 计算准确率
        correct = (y_pred == test_y).sum()
        accuracy = correct / len(test_y)
        accuracies.append(accuracy)

    # 返回每次验证的准确率和平均准确率
    mean_accuracy = np.mean(accuracies)
    return accuracies, mean_accuracy


def custom_k_fold_cross_validation_np(X, y, model, k_folds=5):
    fold_size = len(X) // k_folds
    accuracies = []

    for i in range(k_folds):
        # 划分训练集和测试集
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        test_X = X[test_start:test_end]
        test_y = y[test_start:test_end]

        train_X = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        train_y = np.concatenate((y[:test_start], y[test_end:]), axis=0)

        # 在训练集上训练你的机器学习模型
        model.fit(train_X, train_y)

        # 在测试集上进行预测
        y_pred = model.predict(test_X)

        # 计算准确率
        correct = np.sum(y_pred == test_y)
        accuracy = correct / len(test_y)
        accuracies.append(accuracy)

    # 返回每次验证的准确率和平均准确率
    mean_accuracy = np.mean(accuracies)
    return accuracies, mean_accuracy


# ---------- k = 3, 5, 10 ---------- #
for k in [3, 5, 10]:
    print(f"-------------- k = {k} --------------")
    for a_model in [
            model.Perceptron(learning_rate=0.01, n_iterations=1000),
            model.KNN(k=5)]:

        accuracies, mean_accuracy = custom_k_fold_cross_validation_pd(
            X_train, y_train, a_model, k_folds=k)
        print(f"----- {a_model} -----")
        for i, accuracy in enumerate(accuracies):
            print(f"Fold {i+1} Accuracy: {accuracy}")
        print(f"Mean Accuracy: {mean_accuracy}")

    for a_model in [
            model.NaiveDecisionTree(max_depth=7),
            model.PrunedDecisionTree(max_depth=7, min_samples_split=2)]:

        accuracies, mean_accuracy = custom_k_fold_cross_validation_np(
            X_train.values, y_train.values, a_model, k_folds=k)
        print(f"----- {a_model} -----")
        for i, accuracy in enumerate(accuracies):
            print(f"Fold {i+1} Accuracy: {accuracy}")
        print(f"Mean Accuracy: {mean_accuracy}")


# ---------- Vote ---------- #
# 算CV準確度
start_time_overall = time.time()
acc_PER, mean_acc_PER = custom_k_fold_cross_validation_pd(
    X_train, y_train, model.Perceptron(learning_rate=0.01, n_iterations=1000), k_folds=5)
print(f"acc_per: {acc_PER}, mean_acc_PER: {mean_acc_PER}")
acc_KNN, mean_acc_KNN = custom_k_fold_cross_validation_pd(
    X_train, y_train, model.KNN(k=5), k_folds=5)
print(f"acc_KNN: {acc_KNN}, mean_acc_KNN: {mean_acc_KNN}")
acc_NDT, mean_acc_NDT = custom_k_fold_cross_validation_np(
    X_train.values, y_train.values, model.NaiveDecisionTree(max_depth=7), k_folds=5)
print(f"acc_NDT: {acc_NDT}, mean_acc_NDT: {mean_acc_NDT}")
acc_PDT, mean_acc_PDT = custom_k_fold_cross_validation_np(
    X_train.values, y_train.values, model.PrunedDecisionTree(max_depth=7, min_samples_split=2), k_folds=5)
print(f"acc_PDT: {acc_PDT}, mean_acc_PDT: {mean_acc_PDT}")


# 訓練train 看test結果
start_time = time.time()
PER = model.Perceptron(learning_rate=0.01, n_iterations=1000)
PER.fit(X_train, y_train)
y_pred_perceptron = PER.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"模型推断时间: {inference_time} 秒")

start_time = time.time()
KNN = model.KNN(k=5)
KNN.fit(X_train, y_train)
y_pred_KNN = KNN.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"模型推断时间: {inference_time} 秒")

start_time = time.time()
NDT = model.NaiveDecisionTree(max_depth=7)
NDT.fit(X_train.values, y_train.values)
y_pred_NDT = NDT.predict(X_test.values)
end_time = time.time()
inference_time = end_time - start_time
print(f"模型推断时间: {inference_time} 秒")

start_time = time.time()
PDT = model.PrunedDecisionTree(max_depth=7, min_samples_split=2)
PDT.fit(X_train.values, y_train.values)
y_pred_PDT = PDT.predict(X_test.values)
end_time = time.time()
inference_time = end_time - start_time
print(f"模型推断时间: {inference_time} 秒")

# 模型的平均准确性
model_accuracies = [mean_acc_PER, mean_acc_KNN, mean_acc_NDT, mean_acc_PDT]
print(model_accuracies)

# 计算每个模型的权重
model_weights = [accuracy / sum(model_accuracies)
                 for accuracy in model_accuracies]
print(model_weights)

# 假设你有四个模型的预测结果 stored in model_predictions
model_predictions = [y_pred_perceptron,
                     y_pred_KNN,
                     y_pred_NDT,
                     y_pred_PDT]

# 合并模型的预测结果
final_predictions = np.zeros_like(model_predictions[0])


for i, model_prediction in enumerate(model_predictions):
    model_prediction = np.array(model_prediction)
    final_predictions = (final_predictions +
                         model_prediction * model_weights[i]).astype(np.int32)

# 最终的预测结果
final_predictions = np.where(final_predictions >= 0.5, 1, 0)
print(final_predictions)

accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy: {accuracy}")


# 输出分类报告
print("Classification Report:\n", classification_report(y_test, final_predictions))
end_time_overall = time.time()
inference_time_overall = end_time_overall - start_time_overall
print(f"模型推断时间: {inference_time_overall} 秒")

print(
    f"acc_Pepceptron in test dataset: {accuracy_score(y_test,y_pred_perceptron)}")
print(f"acc_KNN in test dataset: {accuracy_score(y_test,y_pred_KNN)}")
print(f"acc_NDT in test dataset: {accuracy_score(y_test,y_pred_NDT)}")
print(f"acc_PDT in test dataset: {accuracy_score(y_test,y_pred_PDT)}")

# ---------- test ---------- #


# 創建數據結構，每一列代表不同組的測量值
data = pd.DataFrame({
    'PER': acc_PER,
    'KNN': acc_KNN,
    'NDT': acc_NDT,
    'PDT': acc_PDT
})

# 執行Friedman測試
statistic, p_value = friedmanchisquare(*data.values)

# 印出Friedman測試結果
print("Friedman統計值:", statistic)
print("p值:", p_value)

# 進行統計檢定
alpha = 0.05
if p_value < alpha:
    print("組之間存在統計上的差異")
    # 執行Nemenyi多重比較檢定
    nemenyi_results = posthoc_nemenyi(data)
    print("Nemenyi多重比較檢定結果:\n", nemenyi_results)
else:
    print("組之間沒有統計上的差異")

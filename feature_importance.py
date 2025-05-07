import definition_model as model
import data_processing as df
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import shap

X = df.df.drop("is_claim", axis=1)
y = df.df.loc[:, "is_claim"]

X_train, X_test, y_train, y_test = model.split_data(X, y)


# ----------- Perceptron ----------- #
perceptron = model.Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_train, y_train)

weights = perceptron.weights

feature_weights = [(feature_name, weight)
                   for feature_name, weight in zip(X.columns, weights)]

feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

top_n = 5
selected_feature_weights = feature_weights[:top_n]

print(f"TOP {top_n} important features:")
for feature_name, weight in selected_feature_weights:
    print(f"{feature_name}: {weight}")

selected_features = [feature_name[0]
                     for feature_name in selected_feature_weights]

perceptron.fit(X_train.loc[:, selected_features], y_train)
y_pred = perceptron.predict(X_test.loc[:, selected_features])

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(
    y_test, y_pred, zero_division=1))


# ----------- Decision Tree ----------- #
NDT = model.NaiveDecisionTree(max_depth=5)
NDT.fit(X_train.values, y_train.values)

top_features = NDT.top_n_important_features(X, n=5)
print("前五個重要的特徵及其特徵名稱:", top_features)
top_feature_names = [feature[0] for feature in top_features]
print("前五個重要的特徵名稱:", top_feature_names)

# re-train with important features
NDT.fit(X_train.loc[:, top_feature_names].values, y_train.values)
y_pred = NDT.predict(X_test.loc[:, top_feature_names].values)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# ----------- SHAP ----------- #
# PER
perceptron = model.Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_train, y_train)

explainer = shap.Explainer(perceptron, X_train.to_numpy())
shap_values = explainer.shap_values(
    X_test.to_numpy())  # 這裡的X_test是測試數據
shap.summary_plot(shap_values, X_test)


# NDT
NDT = model.NaiveDecisionTree(max_depth=5)
NDT.fit(X_train.values, y_train.values)


class SHAPWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model.predict(x)


shap_model = SHAPWrapper(NDT)
explainer = shap.Explainer(shap_model, X_train.values)
shap_values = explainer(X_train.values)
shap.summary_plot(shap_values, X_train.values, feature_names=X_train.columns)


# ----------- New Feature ---------- $
X['volumn'] = X['height'] * X['width'] * X['length']

newX = X.loc[:, ['volumn', 'height', 'width', 'length',
                 'age_of_car', 'torque to rpm ratio', 'area_cluster_C20', 'population_density',
                 'gross_weight', 'displacement', 'policy_tenure', 'ncap_rating', 'airbags', 'make', 'area_cluster_C15', 'cylinder', 'is_adjustable_steering', 'is_day_night_rear_view_mirror', 'fuel_type_Diesel', 'transmission_type_Manual', 'area_cluster_C8', 'is_tpms', 'is_ecw', 'engine_type_K10C',
                 'is_parking_sensors',
                 'is_power_door_locks', 'area_cluster_C18', 'engine_type_1.5 Turbocharged Revotron', 'is_driver_seat_height_adjustable', 'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'engine_type_K Series Dual jet', 'is_central_locking', 'is_power_steering', 'engine_type_G12B', 'area_cluster_C15']]


X_train, X_test, y_train, y_test = model.split_data(newX, y)

# perceptron
perceptron = model.Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(
    y_test, y_pred, zero_division=1))

# NDT
NDT = model.NaiveDecisionTree(max_depth=5)
NDT.fit(X_train.values, y_train.values)
y_pred = NDT.predict(X_test.values)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(
    y_test, y_pred, zero_division=1))

import pandas as pd
from sklearn.metrics import logloss
from sklearn.preprocessing import LabelEncoder

# train data
x_train = pd.read_csv("./input/otto-group-product-classification-challenge/train.csv")
y = x_train["target"]
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y).astype("int")

#NNs
oof_pred_nn = pd.read_csv("oof_stacking_nn_0314_2.csv").drop(["id"], axis = 1)

# lightgbm 
oof_pred_lgb = pd.read_csv("oof_stacking_lgb_0314.csv").drop(["id"], axis = 1)

from scipy.optimize import minimize
def log_loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction

    return log_loss(y, final_prediction)

predictions = []
predictions.append(oof_pred_nn)
predictions.append(oof_pred_lgb)

starting_values = [0.5] * len(predictions)
cons = ({"type": "eq", "fun":lambda w: 1 - sum(w)})
# our weights are bound in [0, 1]
bounds = [(0, 1)] * len(predictions)

result = minimize(log_loss_func, starting_values, method = "SLSQP", bounds = bounds, constraints = cons)

print("Ensemble Score: {best_score}".format(best_score = result["fun"]))
print("Best weights : {weights}".format(weights = result["x"]))
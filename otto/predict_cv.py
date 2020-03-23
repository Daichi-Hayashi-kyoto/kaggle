from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from time import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
'''
NNs用のpredict_cv

'''
def predict_cv_nn(model, x_train, y_train, x_test, test_predict = True):
    val_preds = []
    preds_test = []
    va_idxes = []
    scores = []

    #kf = KFold(n_splits = 4, shuffle = True, random_state = 42)
    kf = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 884)
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for tr_idx, va_idx in kf.split(x_train, y_train):
        
        y_train_nn = to_categorical(y_train)
        #print(y_train)
        y_train_nn = pd.DataFrame(y_train_nn).astype("int") 
        #print(y_train)
        
        tr_x, val_x = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        tr_y, val_y = y_train_nn.iloc[tr_idx], y_train_nn.iloc[va_idx]
        
        start_time = time()
        model.trainer(tr_x, tr_y, val_x, val_y)
        val_pred = model.predict(val_x)
        
        score = log_loss(val_y, val_pred)
        scores.append(score)
        val_preds.append(val_pred)
        if test_predict:
            pred_test = model.predict(x_test)  # testデータの予測 時間かかるよ...
            preds_test.append(pred_test)
            
        va_idxes.append(va_idx)
        end_time = time()
        
        print("1回のCVにかかる時間は{}分です．".format((end_time - start_time)/60))

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    
    # [ array([...]), array([***])] --> [array([... ***])
    va_idxes = np.concatenate(va_idxes)  
    val_preds = np.concatenate(val_preds, axis=0)
    order = np.argsort(va_idxes)   # va_idxesをsortした時に、ソート前のindex値をorderは保持している
    
    pred_train = val_preds[order]  # index順になる
    # テストデータに対する予測値の平均をとる
    
    if test_predict:
        preds_test = np.mean(preds_test, axis=0)
    
    cv_scores = np.mean(scores)
    
    print("val lossは{}です。".format(cv_scores))

    return pred_train, preds_test, cv_scores, model



'''
LightGBM用のCV回すやつ.
'''

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from time import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, x_train, y_train, x_test, test_predict = True):
    val_preds = []
    preds_test = []
    va_idxes = []
    scores = []

    #kf = KFold(n_splits = 4, shuffle = True, random_state = 42)
    kf = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 884)
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for tr_idx, va_idx in kf.split(x_train, y_train):
        
        tr_x, val_x = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        tr_y, val_y = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        
        start_time = time()
        model.trainer(tr_x, tr_y, val_x, val_y)
        val_pred = model.predict(val_x)
        
        score = log_loss(val_y, val_pred)
        scores.append(score)
        val_preds.append(val_pred)
        if test_predict:
            pred_test = model.predict(x_test)  # testデータの予測 時間かかるよ...
            preds_test.append(pred_test)
            
        va_idxes.append(va_idx)
        end_time = time()
        
        print("1回のCVにかかる時間は{}分です．".format((end_time - start_time)/60))

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    
    # [ array([...]), array([***])] --> [array([... ***])
    va_idxes = np.concatenate(va_idxes)  
    val_preds = np.concatenate(val_preds, axis=0)
    order = np.argsort(va_idxes)   # va_idxesをsortした時に、ソート前のindex値をorderは保持している
    
    pred_train = val_preds[order]  # index順になる OOF
    # テストデータに対する予測値の平均をとる
    
    if test_predict:
        preds_test = np.mean(preds_test, axis=0)
    
    cv_scores = np.mean(scores)
    
    print("val lossは{}です。".format(cv_scores))

    return pred_train, preds_test, cv_scores, model
import lightgbm as lgb

class Model_LightGBM:

    def __init__(self):
        self.model = None

    def trainer(self, x_train, y_train, x_val, y_val):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
        lgb_train = lgb.Dataset(self.x_train, self.y_train)
        lgb_val = lgb.Dataset(self.x_val, self.y_val, reference = lgb_train)

       # LightGBM parameters(chose optuna best)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': {'multi_logloss'},
            'num_class': 9,
            'learning_rate': 0.001,
            "max_depth": -1,
            'lambda_l1': 5.292814363841276,
             'lambda_l2': 3.5754242135119405e-08,
             'num_leaves': 400,
             'feature_fraction': 0.4281527499588857,
             'bagging_fraction': 0.6022707815683139,
             'bagging_freq': 2,
             'min_child_samples': 100, 
            'verbose': 1
            }

        gbm = lgb.train(params, 
                        lgb_train, 
                        valid_sets = lgb_val, 
                        num_boost_round = 500000, 
                        early_stopping_rounds = 100)

        self.model = gbm

    def predict(self, x_test):

        y_pred = self.model.predict(x_test, num_iteration = self.model.best_iteration)
        return y_pred
    
    def feature_importance(self):
        return self.model.feature_importance(importance_type = "gain")


# NNsによるモデル
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras.layers import BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.layers.advanced_activations import PReLU
import math

class Model_NNs:

    def __init__(self):
        self.model = None
        #self.p = p
        

    def trainer(self, x_train, y_train, x_val, y_val):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
         # 中間層 3層
        feature_num = x_train.shape[1]
        model = models.Sequential()
        #model.add(layers.Dropout(0.15))
        model.add(layers.Dense(512, input_dim = feature_num, activation = "relu"))
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(9, activation = "softmax"))
    
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 15, verbose=0, mode='auto')
        check_point = ModelCheckpoint(
                                  filepath = 'best_weights.h5',
                                  monitor = 'val_loss',
                                  verbose = 1,
                                  save_best_only = True,
                                  save_weights_only= True,
                                  mode='min',
                                  period=1)

        # 学習スタート
        model.compile(optimizer = optimizers.SGD(lr=0.004, decay=1e-7, momentum=0.99, nesterov=True), loss = losses.categorical_crossentropy, metrics = [metrics.categorical_crossentropy])
        history = model.fit(self.x_train, self.y_train , epochs = 260, batch_size = 128, validation_data = (self.x_val, self.y_val), callbacks = [early_stop, check_point])
        
        self.model = model


    def predict(self, x_test):
        self.model.load_weights("best_weights.h5")
        y_pred = self.model.predict(x_test)
        return y_pred


'''
Extra Random Tree
'''

from sklearn.ensemble import ExtraTreesClassifier

class Model_extra_tree:
    
    def __init__(self):
        self.model = None
        
    def trainer(self, x_train, y_train, x_val, y_val):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
        # Optunaによるベストパラメータ
        param = {"n_estimators": 732, "min_samples_split": 4, 'min_sample_leaf': 2, 'max_leaf_nodes': 12367}
        
        clf = ExtraTreesClassifier(n_estimators = 732, min_samples_split = 4, min_samples_leaf = 2, max_leaf_nodes = 12367)
        clf.fit(self.x_train, self.y_train)
        self.model = clf
        
    def predict(self, x_test):
        
        y_pred = self.model.predict_proba(x_test)
        return y_pred


'''
k-NN
'''
from sklearn.neighbors import KNeighborsClassifier
class model_knn:
    
    def __init__(self, k):
        self.model = None
        self.k = k
        
    def trainer(self, x_train, y_train, x_val, y_val):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
        knn = KNeighborsClassifier(n_neighbors = self.k)
        knn.fit(self.x_train, self.y_train)  
        self.model = knn
        
    def predict(self, x_test):
        
        y_proba = self.model.predict_proba(x_test)
        return y_proba
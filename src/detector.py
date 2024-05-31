import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

class Detector:
    def __init__(self, mod='rfr', df=None, t_size=0.3, r_state=42):
        self.mod = mod
        if df is not None:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            self.X = df.drop("Is Laundering", axis=1)
            self.y = df['Is Laundering']

            cats = df.select_dtypes(exclude=np.number).columns.tolist()
            self.X.loc[:, cats] = OrdinalEncoder().fit_transform(self.X[cats])

            X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=t_size,random_state=r_state)
            # scaler = StandardScaler()
            # scaler.fit(X_train)
            self.model = self.train_model(mod, X_train, y_train)
            print("Test set score: %f" % self.model.score(X_test, y_test))
            self.check_performance(X_test, y_test)
        else:
            if mod == 'rfr':
                with open("src/pickles/rfr.pkl", "rb") as f:
                    self.model = pkl.load(f)
            elif mod == 'mlp':
                with open("src/pickles/mlp.pkl", "rb") as f:
                    self.model = pkl.load(f)

    

    def train_model(self, mod,  X_train, y_train):
        if mod == 'rfr':
            model = RandomForestRegressor(verbose=1)
            model.fit(X_train, y_train)
        else:
            model = RandomForestRegressor(verbose=1)
            model.fit(X_train, y_train)

            # plot_learning_curve(model,"Crash Learning", X_train, y_train)

            print("Training set score: %f" % model.score(X_train, y_train))

        return model
    
    def check_performance(self, X_test, y_test):
        pred = {} 
        y_pred = self.model.predict(X_test)
        pred['Random Forest Regressor'] = y_pred
        acc= {} 

        for name, y_pred in pred.items():
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            acc[name] = r2
            # accur = accuracy_score(y_test, y_pred)
            print(f"Results for {name} : ")
            print (f"Mean Absolute Error : {mae}")
            print (f"Mean Square Error : {mse}")
            print(f"R2 Score : {r2}")
            # print(f"Accuracy Score : {accur}")

    def check_df(self, df):
        if self.mod == 'rfr':
            ch_df = df.copy()
            cats = ch_df.select_dtypes(exclude=np.number).columns.tolist()
            ch_df.loc[:, cats] = OrdinalEncoder().fit_transform(ch_df[cats])

            return self.model.predict(ch_df) > 0.75
        
        elif self.mod == 'mlp':
            ch_df = df.copy()
            ch_df = ch_df.loc[:, ~ch_df.columns.str.contains('^Unnamed')]
            cats = ch_df.select_dtypes(exclude=np.number).columns.tolist()
            ch_df.loc[:, cats] = OrdinalEncoder().fit_transform(ch_df[cats])
            scaler = StandardScaler()
            ch_df = scaler.fit_transform(ch_df)

            return self.model.predict(ch_df) > 0.75
    
    def save_model(self, name):
        with open(name, 'wb') as f:
            pkl.dump(self.model, f)

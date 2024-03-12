import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from math import ceil
from sklearn.metrics import mean_squared_error
import random
from sklearn.ensemble import GradientBoostingRegressor

# Cleveland's LOWESS Algorithm

def cleveland_lowess(x, y, f=2./3., iter=3):
    n = len(x)
    r = int(ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    # w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    x = np.array(x)
    w = np.clip(np.abs((x[:, np.newaxis] - x[np.newaxis, :]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest

# Axial Velocity smoothing using LOWESS

def smooth_y(dataframe, column, frac):
    X = dataframe['z']
    y = dataframe[column]
    
    yest = cleveland_lowess(X, y, f = frac, iter = 3)
    
    plt.figure(figsize = (16,10), dpi = 300)
    sns.scatterplot(x = 'z', y = column, data = dataframe)
    plt.plot(dataframe['z'], yest, color = 'red', linestyle = '--', label = f'frac = {frac}')
    plt.xlabel('z')
    plt.ylabel(column)
    plt.grid()
    plt.legend()

#Function for comparing two trnedlines after application of LOWESS 

def smooth_y_update(dataframe, column, frac):    
    X = dataframe['z']
    y = dataframe[column]
    
    s = column.split(':')[1]
    s = s.strip()
    
    yest1 = cleveland_lowess(X, y, f = 0.35, iter = 3)
    yest2 = cleveland_lowess(X, y, f = frac, iter = 3)
    
    plt.figure(figsize = (16,10), dpi = 300)
    sns.scatterplot(x = 'z', y = column, data = dataframe)
    plt.plot(dataframe['z'], yest1, color = 'green', linestyle = '--', label = 'frac = 0.35(old)')
    plt.plot(dataframe['z'], yest2, color = 'red', label = f'frac = {frac}(updated)')
    plt.xlabel('z')
    plt.ylabel(column)
    plt.grid()
    plt.legend()
    plt.title('Trendlines with data')
    plt.show()
    
    plt.figure(figsize = (16,10), dpi = 300)
    plt.plot(dataframe['z'], yest1, color = 'green', linestyle = '--', label = 'frac = 0.35(old)')
    plt.plot(dataframe['z'], yest2, color = 'red', label = f'frac = {frac}(updated)')
    plt.xlabel('z')
    plt.ylabel(column)
    plt.grid()
    plt.legend()
    plt.title('Trendline Comparison (Old v. Updated)')
    plt.show()

#Function to prepare data for axial velocity prediction

def make_axial_velocity_data(dataframe):
    column_names = ['z', 'Q', 'RPM', 'axial_velocity']
    data = pd.DataFrame(columns = column_names)

    for column in dataframe.columns[1:]:
        s = column.split(':')[1]
        s = s.split('_')
        q = s[0].strip()
        rpm = s[1]
        
        q_value = [q] * len(dataframe)
        rpm_value = [rpm] * len(dataframe)
        
        data_dict = {'z':dataframe['z'],
            'Q':q_value,
            'RPM':rpm_value,
            'axial_velocity':dataframe[column]}
        
        temp = pd.DataFrame(data_dict)
        
        data = pd.concat([data, temp], axis = 0)
        
    data['Q'] = data['Q'].astype(int)
    data['RPM'] = data['RPM'].astype(float)
    data = data.reset_index(drop = True)

    return data

#Function to check performance of axial velocity prediction model

def check_performance(model, X_train, y_train, X_valid, y_valid, valid_cols):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_nrmse = np.sqrt(mean_squared_error(y_train, train_pred))/(np.max(y_train) - np.min(y_train))
    valid_pred = model.predict(X_valid)
    valid_nrmse = np.sqrt(mean_squared_error(y_valid, valid_pred))/(np.max(y_valid) - np.min(y_valid))
    
    print(f'The train NRMSE is: {train_nrmse}')
    print(f'The valid NRMSE is: {valid_nrmse}')
    
    plt.plot(X_valid[:int(len(X_valid)/3)]['z'], y_valid[:int(len(X_valid)/3)], color = 'blue', label = 'Actual')
    plt.plot(X_valid[:int(len(X_valid)/3)]['z'], valid_pred[:int(len(X_valid)/3)], color = 'red', label = 'Predicted')
    plt.title(valid_cols[0])
    plt.legend()
    plt.show()

    plt.plot(X_valid[int(len(X_valid)/3):int(2*len(X_valid)/3)]['z'],
             y_valid[int(len(X_valid)/3):int(2*len(X_valid)/3)], color = 'blue', label = 'Actual')
    plt.plot(X_valid[int(len(X_valid)/3):int(2*len(X_valid)/3)]['z'],
             valid_pred[int(len(X_valid)/3):int(2*len(X_valid)/3)], color = 'red', label = 'Predicted')
    plt.title(valid_cols[1])
    plt.legend()
    plt.show()

    plt.plot(X_valid[int(2*len(X_valid)/3):]['z'], y_valid[int(2*len(X_valid)/3):], color = 'blue', label = 'Actual')
    plt.plot(X_valid[int(2*len(X_valid)/3):]['z'], valid_pred[int(2*len(X_valid)/3):],
             color = 'red', label = 'Predicted')
    plt.title(valid_cols[2])
    plt.legend()
    plt.show()

#Function to plot parity plots for axial velocity prediction

def plot_parity(model, X_train, X_valid, y_valid, valid_cols):

    valid_pred = model.predict(X_valid)

    subset_size = 181

    num_subsets = len(X_train) // subset_size
    
    for i, vel in zip(range(num_subsets), valid_cols):
        # s = vel.split(':')[1]
        # s = s.split('_')
        # q = s[0].strip()
        # rpm = s[1]

        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size

        plt.figure(figsize=(10, 6))

        plt.scatter(y_valid[start_idx:end_idx], valid_pred[start_idx:end_idx], color='blue', alpha=0.6)
        plt.plot([min(y_valid[start_idx:end_idx]), max(y_valid[start_idx:end_idx])],
                 [min(y_valid[start_idx:end_idx]), max(y_valid[start_idx:end_idx])], color='red', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Parity Plot - {vel}')
        plt.show()

#Function to make random predictions for axial velocity

def random_predictions(dataframe, target):
    idx_list = 181 * np.arange(0, 16)

    valid = pd.DataFrame()
    valid_cols = []

    last_idx = -1

    while valid.shape[0] < 181 * 3:
        rand_idx = random.randint(0, 14)
        if rand_idx != last_idx:
            q = str(dataframe.iloc[idx_list[rand_idx]]['Q'])
            rpm = str(dataframe.iloc[idx_list[rand_idx]]['RPM'])
            if target == 'axial_velocity':
                valid_cols.append('ax: ' + q + '_' + rpm)
            else:
                valid_cols.append(q + '_' + rpm)
            valid = pd.concat([valid, dataframe.iloc[idx_list[rand_idx]:idx_list[rand_idx + 1]]], axis=0)
        last_idx = rand_idx

    train = dataframe.drop(valid.index, axis=0)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_valid = valid.drop([target], axis=1)
    y_valid = valid[target]

    model = GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 200, subsample = 0.8, max_features = 2,
                                    min_samples_split = 36, random_state = 42)

    check_performance(model, X_train, y_train, X_valid, y_valid, valid_cols)

#Function to prepare data for diffusion coefficient prediction

def make_diffusion_coefficient_data(dataframe, dc, mode = 0):
    column_names = ['z', 'Q', 'RPM', 'diffusion_coefficient']
    data = pd.DataFrame(columns = column_names)

    if mode == 0:
        q = []
        rpm = []

        for column in dataframe.columns[1:]:
            s = column.split(':')[1]
            s = s.split('_')
            q.append(s[0].strip())
            rpm.append(s[1]) 
            
        data['Q'] = q
        data['RPM'] = rpm
        data['diffusion_coefficient'] = dc
        data = data.drop('z', axis = 1)
        data['Q'] = data['Q'].astype(int)
        data['RPM'] = data['RPM'].astype(float)
        data['RPM'] = data['RPM'].replace(45.1, 45.0)

    if mode == 1:
        diffusion_coefficients = []

        for i in range(len(dc)):
            group_start = i * 181
            group_end = (i + 1) * 181
            
            diffusion_coefficient = (dataframe['axial_velocity'][group_start:group_end] ** 2) * dc[i]
            
            diffusion_coefficients.extend(diffusion_coefficient)

        data = dataframe.copy()
        data['diffusion_coefficient'] = diffusion_coefficients
        data = data.drop('axial_velocity', axis = 1)
        data['RPM'] = data['RPM'].replace(45.1, 45.0)

    return data
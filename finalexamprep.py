# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import math
from scipy.optimize import minimize
# import matplotlib.pyplot as plt


#%% Data

# Set stocks symbols and time frame
stocks_list = 'AAPL K' # tickers seperated by space
start_date = '2009-01-01'
end_date = '2019-01-01'

# Read
data = yf.download(tickers=stocks_list, start=start_date, end=end_date, interval="1mo")['Adj Close'].dropna()


#%% Linear Regression

monthly_returns = np.log(data).diff().dropna()

mean_returns = monthly_returns.mean()

std_returns = monthly_returns.std()

covar_returns = monthly_returns.cov()

corr_returns = monthly_returns.corr()

x = np.array(monthly_returns['AAPL']).reshape(-1,1)
y = np.array(monthly_returns['K'])

model = LinearRegression().fit(x,y)
test = model.score(x,y)

coefficient = model.coef_[0]
intercept = model.intercept_

#%% Portfolio Modelling

Xa = 0.5
Xb = 1 - Xa

proportions = pd.DataFrame(data=[[Xa], [Xb]], columns=["Weights"], index=['AAPL', 'K']).transpose()

t_monthlyreturns = monthly_returns.transpose()

portfolio_returns = proportions.dot(t_monthlyreturns).transpose()

portfolio_var = proportions.dot(covar_returns).dot(proportions.transpose())

portfolio_std = np.sqrt(portfolio_var).iloc[0,0]


# Minimum Variance
def solverapproach(Xa):
    Xb = 1 - Xa
    proportions = np.array([Xa, Xb]).transpose()
    portfolio_variance = proportions.dot(covar_returns).dot(proportions.transpose())
    return portfolio_variance

result = minimize(solverapproach, x0=0, method="nelder-mead")

AAPL_GMVP = result.x[0]
K_GMVP = 1 - AAPL_GMVP


proportions_GMVP = pd.DataFrame(data=[[AAPL_GMVP], [K_GMVP]], columns=["Weights"], index=['AAPL', 'K']).transpose()

GMVP_var = proportions_GMVP.dot(covar_returns).dot(proportions_GMVP.transpose())

GMVP_sigma = GMVP_var ** 0.5

#%% Simulations

def sim_return(mu, delta_t, sigma, Z):
    r = mu * delta_t + sigma * np.sqrt(delta_t) * Z
    return r
    
mu = 0.12
sigma = 0.3
delta_t = 0.0833
p_t0 = 12
months = np.arange(1,13)
prices = np.zeros(12)
prices[0] = p_t0
Z = np.random.randn(12)

for i in range(1,12):
    prices[i] = prices[i-1] * sim_return(mu,delta_t,sigma,Z[i])
    df = pd.DataFrame({"price":prices}, index=months)    

# Simulation of two stocks

mu1 = 0.12
mu2 = 0.15

sigma1 = 0.22
sigma2 = 0.3

#correlation coefficient
rho = 0.5

delta_t = 0.0833

Z1 = np.random.randn(12)
Z3 = np.random.randn(12)
Z2 = rho * Z1 + np.sqrt(1-rho**2) * Z3

returns1 = np.zeros(12)
returns2 = np.zeros(12)
months = np.arange(1,13)

for i in range(12):
    returns1[i] = sim_return(mu1, delta_t, sigma1, Z1[i])
    returns2[i] = sim_return(mu2, delta_t, sigma2, Z2[i])
    
df2 = pd.DataFrame({"stock 1": returns1, "stock 2": returns2, "Z1": Z1, "Z2": Z2}, index=months)

#%% New Simulations

prices=yf.download('AAPL', start='2017-01-03', end='2017-11-20', interval= '1d')['Adj Close']

num_simulations = 1000
num_days = 30

# price is int
df_simulations = pd.DataFrame()

mu = prices.mean()
sigma = prices.std()
last_price = prices[-1]


for x in range(num_simulations): #bigger number goes here
    count = 0
    delta_t = 1/(num_days*num_simulations) # has to happen inside loop
    price_series = []

    #calculate last price
    price = last_price * np.exp(mu * delta_t + sigma * np.sqrt(delta_t) * np.random.randn())    
    price_series.append(price)
    
    for y in range(num_days): #list of prices/array getting calculated here
        if count == 29:
            break
        #copy paste from above but with [count] index
        price = price_series[count] * np.exp(mu * delta_t * sigma * np.sqrt(sigma) * np.random.randn())
        price_series.append(price)
        count += 1
    
    #add to dataframe
    df_simulations[x] = price_series
    mean_sim = df_simulations.mean(axis=1)
#%% Event Study

# Set stocks symbols and time frame
stocks_list = 'AAPL ^GSPC' # tickers seperated by space
start_date = '2006-01-01'
end_date = '2016-01-01'

# Read
data = yf.download(tickers=stocks_list, start=start_date, end=end_date, interval="1d")['Adj Close'].dropna()

monthly_returns = np.log(data).diff().dropna()
monthly_returns.insert(0,column='SN',value=np.arange(len(monthly_returns)))


x = np.array(monthly_returns["AAPL"]).reshape(-1,1)
y = np.array(monthly_returns["^GSPC"])

x_total = x[0:252]
y_total= y[0:252]

x_eventwindow = x[253:258]
y_eventwindow = y[253:258]

model = LinearRegression().fit(x_total,y_total)

r_squared = model.score(x_total,y_total)
alpha = model.intercept_
beta = model.coef_[0]

# calculating expected return, abnormal return, and cumAR
expected_return = np.array(beta*x_eventwindow + alpha)
abnormal_return = np.array((y_eventwindow - expected_return.transpose()).transpose())
cumAR = np.array(np.cumsum(abnormal_return))

#%% Forecasting ARIMA

# from matplotlib import pyplot
import yfinance as yf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import math

series=yf.download('MSFT', start='2021-01-01', end='2021-04-01', intervals="1d")['Adj Close']

pyplot.plot(series) #check if data is stationary or not, if going up and down it is not
# pyplot.show()

autocorrelation_plot(series) #get value of first lag (x-axis) that intersects the range
# pyplot.show()

X = series.values #remove index
size = int(len(X) * 0.66) 
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)): #value is rolling, will run 22 times
    model = ARIMA(history, order=(8,1,0)) #lag, integration=1 (changes to stationary), moving average=0
    
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0] #y prediction
    predictions.append(yhat)
    
    obs = test[t] #observation, original prices (to be compared with yhat predictions)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse) #change p,d,q to get the lowest error value

# plot forecasts against actual outcomes
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()


#%% Envelope Portfolio

#case of 2 assets

# Input: Portfolio weights
x_weights = np.array([0.5, 0.5]).transpose()
y_weights = np.array([0.3, 0.7]).transpose()

# Portfolios Mean Returns
port_x_ret = mean_returns.dot(x_weights)
port_y_ret = mean_returns.dot(y_weights)

# Portfolios variance calculation of n-assets
x_variance = np.dot(x_weights, covar_returns).dot(x_weights.transpose())
y_variance = np.dot(y_weights, covar_returns).dot(y_weights.transpose())

# Portfolios Standard deviation
x_sigma = math.sqrt(x_variance)
y_sigma = math.sqrt(y_variance)

# Covariance(X,Y) same formula as above
covar_xy = np.dot(x_weights, covar_returns).dot(y_weights.transpose())

# Correlation Coefficient (X,Y)
correl_xy = covar_xy / (x_sigma * y_sigma)

#Envelope Portfolios
#Portfolio Z is a portfolio constructed of portfolio X and portfolio Y.
# Input: proportion of X
Xa = 0.3
z_weights = np.array([Xa, 1-Xa]).transpose() #Proportions of X and Y

# Portfolio X Returns
z_returns = z_weights.dot(np.array([port_x_ret, port_y_ret]))

# Portfolio Variance
z_var = z_weights[0] * 2 * x_variance + z_weights[1] * 2 * y_variance + 2 * np.prod(z_weights) * covar_xy

# Portfolio Sigma
z_sigma = math.sqrt(z_var)
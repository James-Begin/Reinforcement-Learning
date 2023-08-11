import numpy as np
import pandas as pd

import tensorflow.python.keras.optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

import sklearn.preprocessing

import datetime as dt

import yfinance as yf

start = dt.datetime(2015,1,1)
end = dt.datetime(2020,1,1)
SPY_Data = yf.download('SPY', start, end)
QQQ_Data = yf.download('QQQ', start, end)
Bond_Data = yf.download('TLT', start, end)

closeprices = pd.DataFrame()

closeprices[0] = SPY_Data['Adj Close']
closeprices[1] = QQQ_Data['Adj Close']
closeprices[2] = Bond_Data['Adj Close']
closeprices.reset_index(drop=True, inplace=True)

print(closeprices)

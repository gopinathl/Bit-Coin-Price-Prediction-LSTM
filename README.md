# Bit-Coin-Price-Prediction-LSTM

## Problem Statement

The bitcoin prices of the higly fluctuating market from the start of 14th February till the end of 24th of February is provided in the dataset. The goal of the project is to identify the patterns in the time-series data and forecast future bitcoin prices. 

## Dataset
The dataset contains more than 86 lakh rows of bitcoin ask, bid and mid prices at that particular timestamp.

1. local_time - Time at which the prices were recorded
2. ask - The price a seller is willing to sell a bitcoin for
3. bid -  The price a buyer is willing to buy a bitcoin for
4. mid_price - Average of both ask and bid prices

The link to the dataset(btc_usd_pricing_data.csv) is given below:  
Dataset : https://drive.google.com/drive/folders/1yqdapRdmluuoHFQeE_6UIJOb75GqRplo?usp=sharing

## Data Preprocessing

The main pre-processing technique used here is converting the data into uniform time intervals. The data initially had missing data for multiple minutes and the difference between  time intervals had high variation. Therefore, a predefined time interval of 200 milliseconds is set using the mean of the difference between the timestamps.

Imputing the missing value with mean/mode of the prices would be irrelevant for bitcin prices. Instead of filling the previous bitcoin price(backward filling) in the data, a function is used to compute a set of values for the steady increase/decrease of the stock prices for the time period of missing prices. This ensures that there won't be a sudden fluctuation even when the data is missing for a few minutes.

## Time based Splitting

The data is split into train and test data using time based splitting on the concept that the future predictions will be made more accurately from the recent prices.

## Model Building and Evaluation

A stacked LSTM model is used to forecast the bitcoin prices. The input to the model requires two components:

1. Dataset - numpy array that will be fed into the model
2. look_back - number of previous time steps to use as input variables to predict the next time period

### Model Architecture 

![image](https://user-images.githubusercontent.com/34036465/111897400-c35d9a00-8a45-11eb-8bde-e238a45a53a1.png)

### Loss Function
![image](https://user-images.githubusercontent.com/34036465/111897460-fef86400-8a45-11eb-95ea-e7f113030187.png)





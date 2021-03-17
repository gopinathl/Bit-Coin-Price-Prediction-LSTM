import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json


class Predict_Price:

    def __init__(self, file_path, model_weight_path, model_json_path, look_back):
        self.file_path = file_path
        self.model_weight_path = model_weight_path
        self.model_json_path = model_json_path
        self.look_back = look_back

    def read_csv(self):
        df = pd.read_csv(file_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def pre_process(self, df):
        df = df.reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['local_time'], format='%Y-%m-%d %H:%M:%S.%f')

        df.drop(['ask', 'bid', 'local_time'], axis=1, inplace=True)

        df = df.resample('200ms', on='datetime').agg({'mid_price': 'mean'})

        mid_price = list(df['mid_price'])
        for i in range(len(mid_price)):
            if np.isnan([mid_price[i]]):
                for j in range(i + 1, len(mid_price)):
                    if not np.isnan(mid_price[j]):
                        break
                mid_price[i:j] = np.linspace(mid_price[i - 1], mid_price[j], j - i + 2)[1:-1]
        df.loc[:, 'mid_price'] = mid_price
        df = df[['mid_price']]

        dataset = df.values
        dataset = dataset.astype('float32')
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)
        testX, testY = self.create_dataset(dataset)
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return df, dataset, testX, testY, scaler

    def load_model(self):

        file = open(model_json_path, 'r')
        loaded = file.read()
        file.close()

        model = model_from_json(loaded)
        model.load_weights('lstm_best_bit_coin_prediction.h5')
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def predict(self, model, testX, testY, scaler):

        testPredict = model.predict(testX)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        return testPredict, testY

    def evaluate(self, testPredict, testY):

        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % testScore)

    def price_movement(self, df, dataset, testPredict):

        pred_price = np.empty_like(dataset.reshape(len(dataset)))
        pred_price[look_back:len(testPredict) + look_back] = testPredict.reshape(len(testPredict))
        df.loc[:, 'pred_price'] = pred_price
        df = df[look_back:-1]  # first 10 rows are skipped since it is used to compute future prices

        price = df['pred_price']
        price_movement = [np.nan for i in range(len(price))]
        for i in range(len(price)):
            if price[i] > price[i - 1]:
                price_movement[i] = 'Upward'
            elif price[i] < price[i - 1]:
                price_movement[i] = 'Downward'
        df['price_movement'] = price_movement  # The rows where there is a change in price are populated
        df['price_movement'] = df['price_movement'].fillna(method='bfill').fillna(
            method='ffill')  # The rest of the rows are populated using the following price movement

        return df

    def save_final_csv(self, df, save_path):
        df.to_csv(save_path, index=True)
        print("Saved final csv file with predicted prices and price movement")


if __name__ == '__main__':
    file_path = r'.\btc_usd_pricing_data.csv'  # give filepath of csv file
    model_weight_path = r'.\model\lstm_best_bit_coin_prediction.h5'  # give path of model weight
    model_json_path = r'.\model\model.json'  # give path of model.json
    look_back = 10  # number of previous time steps to use as input variables to predict the next time period

    pred_obj = Predict_Price(file_path, model_weight_path, model_json_path, look_back)
    df = pred_obj.read_csv()
    df, dataset, testX, testY, scaler = pred_obj.pre_process(df)
    model = pred_obj.load_model()
    testPredict, testY = pred_obj.predict(model, testX, testY,scaler)  # prices stored in testPredict
    pred_obj.evaluate(testPredict, testY)
    df = pred_obj.price_movement(df, dataset, testPredict)

    save_path = r'.\btc_usd_pricing_data_predicted.csv'
    pred_obj.save_final_csv(df,save_path)

# fit an ARIMA model and plot residual errors
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
import os
import pickle

OUTPUT_DIR = "/opt/ml/model/"

# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

 
def listdirs(rootdir):
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)
            listdirs(d)
            
if __name__=="__main__":
    
    print("Executing main...")
    rootdir = '/opt/ml/'
    listdirs(rootdir)

    series = read_csv('/opt/ml/input/data/training/shampoo.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
    series.index = series.index.to_period('M')


    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()

    # summary of fit model
    print(model_fit.summary())
    
    
    print("Saving model....")
    #model_fit.save(OUTPUT_DIR + "model.h5") 
    path = os.path.join(OUTPUT_DIR, "model.pickle")
    print(f"saving to {path}")
    with open(path,'wb') as p_file:
        pickle.dump(model_fit, p_file)
    
    listdirs(OUTPUT_DIR)
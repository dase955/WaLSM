import pandas as pd
import lightgbm
import numpy
import sys
from io import StringIO

# dataset: train dataset path
# output: saved model path
def train(dataset: str, output: str):
    # sys.stdout = StringIO()
    df = pd.read_csv(dataset)
    y = df['Target']
    X = df.drop(columns=['Target'])
    clf = lightgbm.LGBMClassifier(n_estimators=2, num_leaves=16, max_depth=8)
    clf.fit(X, y)
    # val_pred = clf.predict(X_test)
    clf.booster_.save_model(output)
    
# model_file: saved model path
# data: predicted data point
'''
def predict_one(model_file: str, data: list[int]):
    # df = pd.read_csv("lgb.csv")
    # y = df['Target']
    # X = df.drop(columns=['Target'])
    # print(model_file)
    assert type(data) is list 
    data = pd.DataFrame(data).T
    model = lightgbm.Booster(model_file=model_file)
    result = model.predict(data)
    # print(result)
    return numpy.argmax(result[0])    
'''    

# model_file: saved model path
# data: predicted data batch
def predict(model_file: str, datas: list[list[int]]):
    # df = pd.read_csv("lgb.csv")
    # y = df['Target']
    # X = df.drop(columns=['Target'])
    # print(model_file)
    assert type(datas) is list 
    datas = pd.DataFrame(datas)
    model = lightgbm.Booster(model_file=model_file)
    results = model.predict(datas)
    # print(result)
    # print(results)
    return [ numpy.argmax(result) for result in results ]    


if __name__ == '__main__':
    train()
    predict()

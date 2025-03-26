import pandas as pd
import lightgbm
import numpy
import math

model_path = '/pg_wal/ycc/'
# model_path = ''

class LGBModel():
    def __init__(self) -> None:
        self.__model = None
        # one unit is 4 bits-per-key, class = 2 mean bits-per-key = 4 * 2 = 8
        # the default bits-per-key value of previous benchmark is 10
        self.__default_class = 4
        self.__bits_per_key = 4 # bits_per_key for one filter unit
        self.__num_probes = math.floor(self.__bits_per_key * 0.69) # 4 * 0.69 = 2.76 -> 2
        self.__rate_per_unit = math.pow(1.0 - math.exp(-self.__num_probes/self.__bits_per_key), self.__num_probes) # false positive rate of one unit
        self.__cost_rate_line = 0.10 # we can torelate deviation that is no more than self.__cost_rate_line * (best I/O cost) (compared to best I/O cost)
        self.__model_name = 'model.txt'
        # self.__host = '127.0.0.1'
        # self.__port = '6666'
        # self.__sock = None
        # self.__server = None
        # normally, one data row will not exceed 1024 Bytes
        # we will check this out in WaLSM c++ client
        # self.__bufsize = 1024
        # self.__conn = 8
        
    def __evaluate_model(self, X: pd.DataFrame, y: pd.Series, c: pd.Series) -> bool: 
        # if model still work well, return true
        count_list = list(c)
        class_list = list(y)
        
        preds_list = list()
        for i in range(0, len(X)):
            preds_list.append(int(self.predict(pd.DataFrame([X.loc[i]]))))
            
        assert len(count_list) == len(class_list)
        assert len(preds_list) == len(class_list)
        
        best_cost = 0.0
        pred_cost = 0.0
        for i in range(0, len(class_list)):
            best_cost += math.pow(self.__rate_per_unit, class_list[i]) * count_list[i]
            pred_cost += math.pow(self.__rate_per_unit, preds_list[i]) * count_list[i]
        
        # print("best cost : " + str(best_cost) + ", pred cost: " + str(pred_cost))
        return math.fabs((pred_cost-best_cost)/best_cost) < self.__cost_rate_line
        
    def train(self, dataset: str) -> str:
        df = pd.read_csv(dataset)
        y = df['Target']
        c = df['Count'] # used to check I/O cost metric
        X = df.drop(columns=['Target', 'Count'])
        if self.__model is not None and self.__evaluate_model(X, y, c): 
            # still work well
            return
        # clf = lightgbm.LGBMClassifier(min_child_samples=1, n_estimators=1, objective="multiclass")
        clf = lightgbm.LGBMClassifier()
        clf.fit(X, y)
        # if we directly set self.__model = clf, then self.__model always predict class 0
        # we need save clf to txt file, then read this model to init self.__model
        clf.booster_.save_model(model_path + self.__model_name)
        self.__model = lightgbm.Booster(model_file=model_path+self.__model_name)
        # print('load a new model')
        return 'new model trained'
        
    def predict(self, datas: pd.DataFrame) -> str:
        # currently, only support one data row
        assert len(datas) == 1 
        if self.__model is not None:
            result = self.__model.predict(datas)
            return str(numpy.argmax(result[0]))    
        else:
            return str(self.__default_class)
    
    '''      
    def __close(self) -> None:
        if self.__sock is not None:
            self.__sock.close()
    '''  
    
    '''  
    def start(self) -> None:
        self.__sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.__sock.bind((self.__host, self.__port))
        self.__sock.listen(self.__conn)
    '''  
        
    ''' 
    def serve(self) -> None:
        while True:
            client, _ = self.__sock.accept()
            msg = self.__sock.recv(self.__bufsize)
            decoded = lgb_util.parse_msg(msg)
            
            if type(decoded) is str:
                # send client nothing
                self.__train(decoded)
            elif type(decoded) is list:
                decoded = lgb_util.prepare_data(decoded)
                # self.__predict(decoded)
                # send client target class str (like '0' or '1' or ... )
                client.send(self.__predict(decoded))
            else:
                print('msg type unknown, LGBServer exit')
                self.__close()
    '''    
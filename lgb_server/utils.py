from typing import Union
import pandas as pd
import sys

dataset_path = '/pg_wal/ycc/'
# dataset_path = ''

# msg should be like 'dataset1.csv'
def parse_train_msg(msg: str) -> str:
    assert type(msg) is str 
    msg_list = msg.split(' ', -1)
    assert len(msg_list) == 1
    
    return dataset_path + msg_list[0]

# msg should be like '0 4 12345678 2 2345678', consisted of integer and ' '
# reminded that that msg shouldn't end with ' ' or start with ' '
# and every integer should be seperated with single ' '
def parse_pred_msg(msg: str) -> list[int]:
    assert type(msg) is str 
    assert msg[-1] != ' ' and msg[0] != ' '
    msg_list = msg.split(' ', -1)
    return [ int(item) for item in msg_list]

# build predict data row from list[int]
def prepare_data(data: list[int]) -> pd.DataFrame:
    assert type(data) is list and type(data[0]) is int
    datas = pd.DataFrame([data])
    return datas

# socket input should be like 't dataset1.csv' or 'p 0 4 12345678 2 2345678'
def parse_msg(msg: str) -> Union[str, list[int]]:
    assert msg[0] == 't' or msg[0] == 'p'
    assert msg[1] == ' '
    assert msg[2] != ' '
    # print('new message : ' + msg[2:])
    if msg[0] == 't':
        return parse_train_msg(msg[2:])
    elif msg[0] == 'p':
        return parse_pred_msg(msg[2:])
    else:
        return None
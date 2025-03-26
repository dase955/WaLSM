## models

used to be called by rocksdb through c++ Python interface(Python.h)

**deprecated** in latest implement of WaLSM. 

in latest version, We decide to use client-server architecture to train and predict 

## Files

 - lgb.cc: simple c++ demo of calling c++ Python interface

 - lgb.py: simple python func for training or predicting

 - lgb.sh: simple shell for running this c++ demo
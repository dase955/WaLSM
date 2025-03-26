## lgb_server

Because the trouble of LightGBM C++ API, we seperate LightGBM train and predict job to a single socket server (python3.12)

once WaLSM need to train new model or predict for new segments, it need to init socket client, send this socket server message and wait for response if necessary (especially for predict job)
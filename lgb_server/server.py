import utils
import model
import socketserver
import time

clf = model.LGBModel()
host = '127.0.0.1'
port = 9090
bufsize = 1024

class LGBhandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            while True:
                # for example
                # if one compaction happen, the new sstable is consisted of 10000 segments
                # we can divide these segments into some groups, 
                # and prefetch filter units for these segments using multi-thread
                # that means in rocksdb, one client thread may need to 
                # predict class for a group of segments
                # that means we need keep this connection until all segments of this group done
                # use 'while True' and TCP protocol to keep connection
                msg = self.request.recv(bufsize).decode('UTF-8', 'ignore').strip()
                if not msg:
                    break
                
                decoded = utils.parse_msg(msg)
                if type(decoded) is str: # mean it is train msg
                    # send client nothing
                    result = clf.train(decoded).encode('UTF-8')
                    # result is a simple str, just means training end
                    self.request.send(result)
                elif type(decoded) is list: # mean it is pred msg, need to send client predict result
                    # print(decoded)
                    decoded = utils.prepare_data(decoded)
                    # send client target class str (like '0' or '1' or ... )
                    result = clf.predict(decoded).encode('UTF-8')
                    # print(clf.predict(decoded))
                    self.request.send(result)
        except ConnectionResetError:               
            print('one connection close: ' +
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# server = socketserver.ThreadingTCPServer((host,port), LGBhandler)
if __name__ == '__main__':
    print('LGBServer start')
    assert utils.dataset_path == model.model_path
    server = socketserver.ThreadingTCPServer((host,port), LGBhandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nLGBServer end')
    
    # should not end during benchmark
    # print('LGBServer end')
fieldcount=10
fieldlength=1600

recordcount=5000000
operationcount=20000000
workload=com.yahoo.ycsb.workloads.CoreWorkload

readallfields=true

readproportion=0.5
updateproportion=0.5
scanproportion=0
insertproportion=0

requestdistribution=zipfian
zipfianvalue=0.98
maxscanlength=100
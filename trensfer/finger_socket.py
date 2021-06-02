#-*- coding:utf-8 -*-
from socket import *
import numpy as np
num = 4

while True:
    file = "finger"+str(num)+".txt"
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.bind(('', 8080))
    serverSock.listen(1)

    connectionSock, addr = serverSock.accept()

    print(str(addr),'에서 접속이 확인되었습니다.')
    
    data = connectionSock.recv(1024)
    print('받은 데이터 : ', data.decode('utf-8'))
    
    p = data.decode('utf-8')

    pixels = p.split(" ")

    pixel = np.array(pixels)
    #pixel = np.reshape(pixel,(8,8))
    
    f = open(file,'w')
    f.write(str(pixel))
    f.close()

    num = num + 1
    
    serverSock.close()

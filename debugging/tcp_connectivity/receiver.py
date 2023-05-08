import socket

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = ""
port = 3630
print (host)
print (port)
serversocket.bind((host, port))

serversocket.listen(5)
print ('server started and listening')
while 1:
    (clientsocket, address) = serversocket.accept()
    print ("connection found!")

    r='Receieved your message everything works'
    clientsocket.send(r.encode())

    data = clientsocket.recv(1024).decode()
    print (data)

    exit(0)

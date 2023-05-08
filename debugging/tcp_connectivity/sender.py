import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host ="172.22.2.163"
port = 3630
s.connect((host, port))

def ts(str):
   s.send(str.encode())
   data = ''
   data = s.recv(1024).decode()
   print (data)

ts('If you see this message in the receiver side everything works.')
s.close ()
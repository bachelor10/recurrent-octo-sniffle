import os

validfiles = os.listdir(os.getcwd() + '/validation')

for f in os.listdir(os.getcwd() + '/train'):
    if f not in validfiles:
        print(f)
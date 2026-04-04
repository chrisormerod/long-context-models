import os

if os.environ['LOCATION'] == "LAPTOP":
    
    DATA_DIR = "/mnt/c/data"
    MODEL_DIR = "/mnt/c/models"

else:

    DATA_DIR = "/home/ubuntu/data"
    MODEL_DIR = "/home/ubuntu/models"
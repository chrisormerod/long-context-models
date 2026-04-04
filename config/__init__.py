import os

if os.environ['LOCATION'] == "LAPTOP":
    
    DATA_DIR = "/mnt/c/data"
    MODEL_DIR = "/mnt/c/models"
    TMP_DIR = "/mnt/c/tmp"

else:

    DATA_DIR = "/home/ubuntu/data"
    MODEL_DIR = "/home/ubuntu/models"
    TMP_DIR = "/home/ubuntu/tmp"
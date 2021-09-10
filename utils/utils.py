
import os

def create_path(path):

    if not os.path.isdir(path):
        os.makedirs(path)

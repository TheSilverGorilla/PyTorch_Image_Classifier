import shutil
import os
import random

def get_random_filename(path):
    return random.choice(os.listdir(path))
object_list = [...]
new_dir = os.path.join(os.getcwd(),"vehicle_classification_test")
os.chdir(new_dir)
path = "path" #path to directory
for i in object_list:
    object_dir = os.path.join(new_dir, i)
    os.mkdir(object_dir)
    for element in range(30):
        src_dir_path = os.path.join(path, i)
        src_file_path = os.path.join(src_dir_path, get_random_filename(src_dir_path))
        shutil.move(src_file_path, object_dir)

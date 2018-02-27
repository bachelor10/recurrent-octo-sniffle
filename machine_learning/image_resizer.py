from PIL import Image
import os, random


def resize_img(pathname, store_pathname, validation_pathname, new_size, max_entries=100000, validation=0.15):
    count = 0
    for dir_name in os.listdir(pathname):
        sub_dir = pathname + '/' + dir_name
        print(sub_dir)
        for subdir_filename in os.listdir(sub_dir):

            img = Image.open(sub_dir + '/' + subdir_filename)
            img.thumbnail(new_size)

            if random.random() < validation:
                img_dir = validation_pathname + '/' + dir_name
            else:
                img_dir = store_pathname + '/' + dir_name
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            img.save(img_dir + '/' + subdir_filename)
            count += 1

            if count > max_entries: 
                count = 0
                break


resize_img(os.getcwd() + '/train', os.getcwd() + '/train2',  os.getcwd() + '/validation2', [26, 26], max_entries=20000)
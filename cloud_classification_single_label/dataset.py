import argparse
import csv
import os
import random
import shutil
from shutil import copyfile

from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)

    origin_GT_file = csv.DictReader(open(config.origin_GT_path+config.origin_GT_file,  encoding='UTF-8-sig'))
    origin_data_list = [i for i in origin_GT_file]
    data_list = []
    for index in range(len(origin_data_list)):
        filename = origin_data_list[index]['FileName']
        multi_GT = origin_data_list[index]['Code'].split(';')
        for j in multi_GT:
            data = {}
            data['FileName'] = filename
            data['Code'] = j
            data_list.append(data)

    train_list = []
    valid_list = []
    test_file = csv.DictReader(open(config.test_GT_path+'test_label.csv', encoding='UTF-8-sig'))
    test_list = [i for i in test_file]

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = len(test_list)

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    print('\nNum of test set : ', num_test)
    
    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        filename = data_list[idx]['FileName']
        src = os.path.join(config.origin_data_path, filename)
        dst = os.path.join(config.train_path, filename)
        copyfile(src, dst)
        
        train_list.append(data_list[idx])

        printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
    
        train_data_writer = csv.DictWriter(open(config.train_GT_path+'train_label.csv','w'), origin_GT_file.fieldnames)
        train_data_writer.writeheader()
        train_data_writer.writerows(train_list)

    for i in range(num_valid):
        idx = Arange.pop()
        filename = data_list[idx]['FileName']
        src = os.path.join(config.origin_data_path, filename)
        dst = os.path.join(config.valid_path, filename)
        copyfile(src, dst)

        valid_list.append(data_list[idx])
        
        printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)

    valid_data_writer = csv.DictWriter(open(config.valid_GT_path+'valid_label.csv','w'), origin_GT_file.fieldnames)
    valid_data_writer.writeheader()
    valid_data_writer.writerows(valid_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='../cloud_origin_dataset/Train/')
    parser.add_argument('--origin_GT_path', type=str, default='../cloud_origin_dataset/Train_GT/')
    parser.add_argument('--origin_GT_file', type=str, default='Train_label.csv')

    parser.add_argument('--train_path', type=str, default='../cloud_single_label_dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='../cloud_single_label_dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='../cloud_single_label_dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='../cloud_single_label_dataset/valid_GT/')
    parser.add_argument('--test_path', type=str, default='../cloud_single_label_dataset/test/')
    parser.add_argument('--test_GT_path', type=str, default='../cloud_single_label_dataset/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)

import argparse
import csv
import numpy as np
import os
from PIL import Image

def write_multi_label(image_file, SR_multi):
    path = 'test_ans_multi_file.csv'
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        for i in range(len(image_file)):
            SR = []
            #print(SR_multi[i].data)
            for j in range(29):
                if SR_multi[i][j].item() == 1:
                    SR.append(str(j+1))
            data_row = [image_file[i], ';'.join(SR)]
            csv_write.writerow(data_row)

def cal_multi_F1():
    #pre_path = 'test_ans_0.csv'
    #pre_path = 'test_ans_multi_file.csv'
    pre_path = 'submit_example.csv'
    #pre_path = 'test_ans_multi_file_2_2_30.csv'
    GT_path = './new_result/result.csv'
    #GT_path = './new_result/result2.csv'
    #GT_path = './new_result/result3.csv'
    SR_file = csv.DictReader(open(pre_path, encoding="UTF-8-sig"))
    GT_file = csv.DictReader(open(GT_path, encoding="UTF-8-sig"))
    SR_list = [i for i in SR_file]
    GT_list = [i for i in GT_file]
    print(len(SR_list))
    print(len(GT_list))
    num = len(SR_list)
    TP = np.zeros(29)
    FP = np.zeros(29)
    FN = np.zeros(29)
    TN = np.zeros(29)
    precision = np.zeros(29)
    recall = np.zeros(29)
    F1 = 0.
    Acc = 0.
    for i in range(num):
        GT_multi = np.zeros(30)
        SR_multi = np.zeros(30)
        SR = SR_list[i]['type'].split(';')
        GT = GT_list[i]['type'].split(';')
        for pi in SR:
             SR_multi[int(pi)-1] = 1
        for gi in GT:
             GT_multi[int(gi)-1] = 1
        for j in range(29):
            if SR_multi[j] == 1 and GT_multi[j] == 1:
                TP[j] = TP[j] + 1
            if SR_multi[j] == 1 and GT_multi[j] == 0:
                FP[j] = FP[j] + 1
            if SR_multi[j] == 0 and GT_multi[j] == 1:
                FN[j] = FN[j] + 1
            if SR_multi[j] == 0 and GT_multi[j] == 0:
                TN[j] = TN[j] + 1
    for i in range(29):
        precision[i] = float(TP[i])/(float(TP[i]+FP[i]) + 1e-6)
        recall[i] = float(TP[i])/(float(TP[i]+FN[i]) + 1e-6)
        F1 = F1 + 2*recall[i]*precision[i]/(recall[i]+precision[i] + 1e-6)
        Acc = Acc + float(TP[i]+TN[i])/(float(TP[i]+TN[i]+FP[i]+FN[i]) + 1e-6)
    F1 = F1 / 29.0
    Acc = Acc / 29.0
    print('F1: %.4f, Acc_multi: %.4f\n' % (F1, Acc))

def calbase():
    path='./base.csv'
    result = './result.csv'
    fa = 2
    base_file = csv.DictReader(open(path, encoding="UTF-8-sig"))
    base_list = [i for i in base_file]
    num = len(base_list)
    tt = ['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12']
    for i in range(num):
        image_name = base_list[i]['FileName']
        cnt = np.zeros(50)
        for ti in tt:
            base = base_list[i][ti].split(';')
            for bi in base:
                if not bi is None and not bi == '':
                  cnt[int(bi)] = cnt[int(bi)]+1
        with open(result,'a+') as f:
            csv_write = csv.writer(f)
            GT = []
            max_GT = 0
            max_GT_cnt = 0
            for i in range(32):
                if cnt[i] > max_GT_cnt:
                    max_GT = i
                    max_GT_cnt = cnt[i]
                #if cnt[i] > fa:
                #    GT.append(str(i))
            print(max_GT)
            #data_row = [image_name, ';'.join(GT)]
            data_row = [image_name, max_GT]
            csv_write.writerow(data_row)


def calPicturesSize():
    path = '../cloud_origin_dataset/Train/'
    #path = '../cloud_origin_dataset/Test/'
    filenames = os.listdir(path)
    minSize = 10000
    maxSize = 0
    cnt3 = 0
    cnt4 = 0
    cnt0 = 0
    for filename in filenames:
        #print(filename)
        image = Image.open(path+filename)
        tt = image.split()
        if len(tt) == 3:
            cnt3 = cnt3 + 1
        elif len(tt) == 4:
            cnt4 = cnt4 + 1
        else:
            cnt0 = cnt0 + 1
        #print('3:4:other=%d:%d:%d' %(cnt3, cnt4, cnt0))
        if image.size[0] == 90 or image.size[1] == 90:        
            print(filename)
            print(image.size)
        if image.size[0] == 9792 or image.size[1] == 9792:        
            print(filename)
            print(image.size)

        
        minSize = min(minSize, min(image.size[0], image.size[1]))
        maxSize = max(maxSize, max(image.size[0], image.size[1]))
    print('size:%d %d' %(minSize, maxSize))
    print('3:4:other=%d:%d:%d' %(cnt3, cnt4, cnt0))

def cal(config):
    origin_GT_file = csv.DictReader(open(config.origin_GT_path+config.origin_GT_file, encoding="UTF-8-sig"))
    data_list = [i for i in origin_GT_file]
    num = len(data_list)
    GT = np.zeros(30)
    GT_2 = np.zeros(30)
    for i in range(num):
        multi_GT = data_list[i]['Code'].split(';')
        code_num = len(multi_GT)
        GT[code_num] = GT[code_num] + 1
        for j in multi_GT:
            print(j)
            GT_2[int(j)] = GT_2[int(j)] + 1
    for i in range(30):
        print(str(i)+" : "+str(GT[i]))
    for i in range(30):
        print(str(i)+" : "+str(GT_2[i]))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--origin_GT_path', type=str, default='../cloud_origin_dataset/Train_GT/')
    parser.add_argument('--origin_GT_file', type=str, default='Train_label.csv')
    config = parser.parse_args()
    cal_multi_F1()
    #calbase()
    #cal(config)

import os

if __name__ == '__main__':
    version_list = ['Cat-CNN']
    group_list = ['sMCI_pMCI', 'CN_AD']
    for version in version_list:
        for group in group_list:
            os.system(r'python eval.py -v {} -g {}'.format(version, group))
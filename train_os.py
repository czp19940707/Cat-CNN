import json
import os

if __name__ == '__main__':
    # ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15']
    # version_list = ['s1', 's2', 's3', 's4', 's5']
    # version_list = ['z11', 'z12', 'z13', 'z14', 'z15']
    # version_list = ['v1', 'v2', 'v3', 'v4', 'v5']
    # version_list = ['v1', 'v2', 'v3', 'v4', 'v5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15']
    # version_list = ['z12', 'z13', 'z14', 'z15']
    # group_list = ['sMCI_pMCI']
    version_list = ['Cat-CNN']
    group_list = ['CN_AD', 'sMCI_pMCI']
    for version in version_list:
        for g in group_list:
            for i in range(5):
                os.system(r'python train.py -v {} -fold {} -g {}'.format(version, i, g))

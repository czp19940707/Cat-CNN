import os

dict_train = {
    'v1': {
        'normalization': True,
        'modality': 'Pet',
    },
    'Cat-CNN':{
        'normalization': True,
        'modality': 'T1_Pet',
    }
}

data_path = {
    'Pet': r'pet/fgd_Corg/template2rainmask_dof12.nii.gz',
    'T1': r'fs/T1/mri/norm.mgz',
}


def model_select(args):
    from model import Conv5_FC3
    return Conv5_FC3(version=args.v)

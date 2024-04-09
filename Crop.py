import argparse
import nibabel as nib
import os


def crop(npy):
    return npy[50: -50, 50:-50, 20: -50]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='/media/shucheng/MyBook/2024_ADNI_part3/002_S_4229', type=str)
    args = parser.parse_args()

    subject_id = os.path.split(args.i)[-1]
    save_path = f'/media/shucheng/MyBook/DL_dataset/Conventional_CNN/{subject_id}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nii_T1 = nib.load(os.path.join(args.i, 'fs', 'T1', 'mri', 'norm.mgz'))
    img_T1 = nii_T1.get_fdata()
    affine_T1 = nii_T1.affine
    img_T1_crop = crop(img_T1)
    nifti_image_T1 = nib.Nifti1Image(img_T1_crop, affine=affine_T1)
    nib.save(nifti_image_T1, os.path.join(save_path, 'T1_crop.nii.gz'))

    nii_pet = nib.load(os.path.join(args.i, 'pet', 'fgd_Corg', 'template2rainmask_dof12.nii.gz'))
    img_pet = nii_pet.get_fdata()
    affine_pet = nii_pet.affine
    img_pet_crop = crop(img_pet)
    nifti_image_pet = nib.Nifti1Image(img_pet_crop, affine=affine_pet)
    nib.save(nifti_image_pet, os.path.join(save_path, 'Pet_crop.nii.gz'))

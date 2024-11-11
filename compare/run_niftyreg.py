import os
import SimpleITK as sitk

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import visual
import time 

class nifty_reg():
    def __init__(self, config) -> None:
        pass 

    def registration(self, moving, fix):
        moving_path = moving[5]
        fix_path = fix[5]

        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]

        # moving_image_path = "/tmp/moving_image.nii.gz"
        # moving_label_path = "/tmp/moving_label.nii.gz"

        fix_image_path = fix_path["image"]
        fix_label_path = fix_path["label"]  

        # visual.save_image(moving[0], moving_image_path, fix_image_path)
        # visual.save_image(moving[1], moving_label_path, fix_label_path)    

        # dice_origin = eval.dice_val(moving[1], fix[1], 5)
        # print(numpy.mean(dice_origin[1:5]))

        tem_result_path = "/tmp"
        # if not os.path.exists(tem_result_path):
        #     os.makedirs(tem_result_path) 

        warped_image_path = os.path.join(tem_result_path, "nifty_result.nii.gz")
        warped_label_path = os.path.join(tem_result_path, "label_result.nii")

        affine_matrix_path = os.path.join(tem_result_path, "affine_matrix.txt")
        cpp_path = os.path.join(tem_result_path, "ref_template_flo_new_image_nrr_cpp.nii")
        disp_path = os.path.join(tem_result_path, "disp.nii.gz")
        def_path = os.path.join(tem_result_path, "def.nii.gz")

        affine_cmd = f"reg_aladin -ref {fix_image_path} -flo {moving_image_path} -aff {affine_matrix_path} -voff -pad 0"
        reg_cmd = f"reg_f3d -be 0.002  --nmi -aff {affine_matrix_path} -ref {fix_image_path} -flo {moving_image_path} -res {warped_image_path} -cpp {cpp_path} -voff -pad 0"
        transform_cmd = f"reg_transform -ref {fix_image_path} -disp {cpp_path} {disp_path}"
        deformation_cmd = f"reg_transform -ref {fix_image_path} -def {cpp_path} {def_path}"
        resample_cmd = f"reg_resample -aff {affine_matrix_path} -ref {fix_label_path} -flo {moving_label_path} -res {warped_label_path} -trans {cpp_path} -inter 0"

        start_time = time.perf_counter()
        os.system(affine_cmd)
        os.system(reg_cmd)
        os.system(transform_cmd)
        os.system(deformation_cmd)
        os.system(resample_cmd)
        end_time = time.perf_counter()

        result = {}
        result["moved_image"] = sitk.ReadImage(warped_image_path)
        result["moved_label"] = sitk.ReadImage(warped_label_path)
        result["affine_matrix"] = None
        result["svf"] = sitk.ReadImage(disp_path)
        result["deformation"] = sitk.ReadImage(def_path)
        result["run_time"] = end_time - start_time

        # os.remove(moving_image_path)
        # os.remove(moving_label_path)
        os.remove(affine_matrix_path)
        os.remove(warped_image_path)
        os.remove(warped_label_path)
        os.remove(cpp_path)
        os.remove(disp_path)
        os.remove(def_path)


        return result
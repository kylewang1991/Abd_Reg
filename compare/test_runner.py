import os
import sys
import shutil

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import visual
import data_loader
import eval

import run_base_line
import run_sync
import run_niftyreg
import run_convexAdam
import run_model
import argparse
import pandas as pd
import datetime as dt

import numpy as np

import json


supported_method = {
                        "baseline" : run_base_line.baseline_reg,
                        "sync" : run_sync.sync_reg,
                        "niftyreg" : run_niftyreg.nifty_reg,
                        "convex_adam" : run_convexAdam.convexadam_reg,
                        "voxelmorph" : run_model.model_reg,
                        "transmorph" : run_model.model_reg,
                        "affine_mi" : run_model.model_reg,
                        "affine_struct" : run_model.model_reg,
                        "sgmaconv": run_model.model_reg,
                        "sgmatrans_0": run_model.model_reg,
                        "sgmatrans_1": run_model.model_reg,
                        "sgmatrans_2": run_model.model_reg
                        }


def setup_data(config_data, data_type="train"):
    run_intra_registration = True if config_data["exp"] == 'intra' else False

    this_dir_path = os.path.dirname(__file__)
    json_path = os.path.join(this_dir_path, config_data["path"])


    affine_mat_dir = os.path.join(os.path.dirname(__file__), "../", config_data["path_affine_mat"]) if config_data["path_affine_mat"] else None

    if data_type=="train" and not config_data["train_use_mat"]:
        affine_mat_dir = None

    if data_type=="valid" and not config_data["valid_use_mat"]:
        affine_mat_dir = None

    if data_type=="test" and not config_data["test_use_mat"]:
        affine_mat_dir = None

    dataset = data_loader.RegDataSet(json_path, 
                                     intra_patient=run_intra_registration, 
                                     affine_mat=affine_mat_dir, 
                                     type=data_type)

    return dataset

if __name__ == "__main__":

    # Get dataset config 
    with open(os.path.join(os.path.dirname(__file__), "../SGMANet/config/dataset.json"), "r") as f:
        config_dataset_all = json.load(f)

    # Parser input
    parser = argparse.ArgumentParser(description = 'Run compare test')

    parser.add_argument('-m', '--method', dest='method',  choices=supported_method.keys(), help='method to be used')
    parser.add_argument('-d', '--data', dest='dataset', choices=config_dataset_all.keys(), help="dataset to use")
    parser.add_argument('-v', '--visual', dest='visual', action='store_true', help="save the moved image and label for visualization")
    parser.add_argument('-t', '--data_type', dest='data_type', default="test", help="data_type: train, valid, test")
    parser.add_argument('-p', '--model_path', dest='model_path', default=None, help="path to the saved model")
    parser.add_argument('-n', '--mat', dest='mat_path', default=None, help="path to save the linear transformation matrix")
    parser.add_argument('-i', '--index', dest='index', default=None, type=int, help="index of the data to be tested")


    args = parser.parse_args()

    # Set up the data loader
    config_dataset = config_dataset_all[args.dataset]
    dataset = setup_data(config_dataset, args.data_type)


    #create dir to save model data
    now = dt.datetime.now()
    result_dir_data = f"{args.dataset}"
    result_dir_method = f"{args.method}"
    result_dir_time = f"{now:%Y-%m-%d-%H-%M-%S}"
    result_dir = os.path.join(os.path.dirname(__file__), "./results", result_dir_data, result_dir_method, result_dir_time)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)      

    #save config
    config = vars(args)
    config["dataset_param"] = config_dataset
 

    # setup method
    method = supported_method[args.method]  
    method = method(config)

    if hasattr(method,  "model_path"):
        config["model_path"] = method.model_path

    # Run test
    record_note_list = []

    for index, (moving, fix) in enumerate(dataset):

        if args.index != None:
            if index != args.index:
                continue
            
        # Record the path
        moving_path = moving[5]
        fix_path = fix[5]

        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]
        fix_image_path = fix_path["image"]
        fix_label_path = fix_path["label"]

        record_node = {}
        _, record_node["moving_image"] = os.path.split(moving_image_path)
        _, record_node["fix_image"] = os.path.split(fix_image_path)         

        #result["moved_image"]: moved image
        #result["moved_label"]: moved label, C*D*W*H, datatype should be uint8, we will assume it is onehot coded if its channel equals to 8
        #result["affine_matrix"]: 3*4 affine parameter in tuple
        #result["svf"] : shift vector field in tensor
        result = method.registration(moving, fix)

        # Save the result for visualization
        if args.visual:
            _, image_name = os.path.split(moving_image_path)
            image_name = image_name.split('.')[0] + '_' + "image" + ".nii.gz"
            path_to_moved_image = os.path.join(result_dir, image_name)

            _, label_name = os.path.split(moving_label_path)
            label_name = label_name.split('.')[0] + "_" + "label" + ".nii.gz"
            path_to_moved_label = os.path.join(result_dir, label_name)

            _, svf_name = os.path.split(moving_image_path)
            svf_name = svf_name.split('.')[0] + "_" + "svf" + ".nii.gz"
            path_to_svf = os.path.join(result_dir, svf_name)           

            visual.save_image(result["moved_image"], path_to_moved_image, fix_image_path)
            visual.save_image(result["moved_label"], path_to_moved_label, fix_label_path)
            
            if "svf" in result:
                visual.save_deformation(result["svf"], path_to_svf, fix_image_path)

            record_node["moved_image_path"] = image_name
            record_node["moved_label_path"] = label_name

        # calculate dice score
        fixed_onehot = fix[2].unsqueeze(0)
        moved_onehot = eval.label_to_onehot(result["moved_label"]).unsqueeze(0)
        dsc = eval.dsc_one_hot(fixed_onehot, moved_onehot).numpy()
        record_node["dsc_0"] = dsc[0]
        record_node["dsc_1"] = dsc[1]
        record_node["dsc_2"] = dsc[2]
        record_node["dsc_3"] = dsc[3]
        record_node["avg_dsc"] = np.mean(dsc)

        result_string = f'move: {record_node["moving_image"]}, fix: {record_node["fix_image"]}, dsc: {record_node["avg_dsc"]: .2f}'
        # calculate jicobal
        if result["svf"] is not None:
            svf_tensor = eval.disp_def_to_tensor(result["svf"])
            jcob = eval.jacobian_determinant_pytorch(svf_tensor)
            record_node["jcob"] = jcob
            result_string = result_string + f', jcob: {record_node["jcob"]: .6f}'

        if "run_time" in result:
            record_node["run_time"] = result["run_time"]
            result_string += f', run_time: {record_node["run_time"]: .6f}'
        
        record_note_list.append(record_node)

        print(result_string)

        if args.mat_path and args.method in ["ants_affine", "affine_mi", "affine_mine", "affine_struct"]:

            if not os.path.exists(args.mat_path):
                os.makedirs(args.mat_path) 

            _, moving_image_name = os.path.split(moving_image_path)
            _, fix_image_name = os.path.split(fix_image_path)
            moving_image_name = moving_image_name.split('.')[0]
            fix_image_name = fix_image_name.split('.')[0]

            mat_name = fix_image_name + '_' + moving_image_name + '.mat'
            mat_path = os.path.join(args.mat_path, mat_name)

            if type(result["affine_matrix"]) == str:
                shutil.copy(result['fwdtransforms'], mat_path)
            elif type(result["affine_matrix"]) == torch.Tensor:
                torch.save(result["affine_matrix"], mat_path)
            else:
                raise ValueError("Unsupported format!")


    df = pd.DataFrame.from_records(record_note_list)
    summary = df.describe()
    summary = summary.loc[['mean', 'std']]

    # save config
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent = 4)   

    # save to excel
    file_name = f"result_{now:%Y_%m_%d_%H_%M_%S}.xlsx"
    file_name = os.path.join(result_dir, file_name)

    with pd.ExcelWriter(file_name) as writer:  
        df.to_excel(writer, sheet_name='data', index=False, float_format='%.6f')
        summary.to_excel(writer, sheet_name='summary', float_format='%.6f')
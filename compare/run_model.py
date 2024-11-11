import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import sys
import glob
import shutil
from natsort import natsorted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import networks
import time

class model_reg():
    def __init__(self, config) -> None:
        
        if config["model_path"] is None:

            path_to_saved_models = os.path.join(os.path.dirname(__file__), "../SGMANet/models/", config["dataset"], config["method"])
            
            if not os.path.exists(path_to_saved_models):
                raise ValueError(f'no model saved for method: {config["method"]} and dataset: {config["dataset"]}')

            search_dir = os.path.join(path_to_saved_models, '*')
            
            print(f"search model in {search_dir} folder")
            model_lists = natsorted(glob.glob(search_dir))
            path_to_saved_model = model_lists[-1]

            config_file = os.path.join(path_to_saved_model, "config.json")
            model_file = self.get_best_model_from_json(os.path.join(path_to_saved_model, "best"))
        elif os.path.isdir(config["model_path"]):
            path_to_saved_model = config["model_path"]

            config_file = os.path.join(path_to_saved_model, "config.json")
            model_file = self.get_best_model_from_json(os.path.join(path_to_saved_model, "best"))
        elif os.path.isfile(config["model_path"]) and \
            ( config["model_path"].endswith(".pth") or config["model_path"].endswith(".pth.tar")):
            file_path, file_name = os.path.split(config["model_path"])

            config_file = os.path.join(file_path, "../", "config.json")
            model_file = config["model_path"]       

        self.model_path = model_file
        self.model, self.device = self.load_model(config_file, model_file)

    @staticmethod
    def get_best_model_from_json(best_dir):

        best_json_file = os.path.join(best_dir, "best.json")

        if not os.path.exists(best_json_file):
            files = glob.glob(os.path.join(best_dir, '*'))
            if len(files) == 1:
                return files[0]

        with open(best_json_file, 'r') as file:
            best_record = json.load(file)
        
        if "best_list" in best_record:
            record = best_record["best_list"]
            record.sort(key=lambda n: n["score"])
            best_record = record[-1]

        print(f"best record in eval: {best_record}")

        # Get best model path
        if os.path.exists(os.path.join(best_dir, "best.pth")):
            model_file = os.path.join(best_dir, "best.pth")
        else:
            model_file = os.path.join(best_dir, f'best_{best_record["epoch"]:04d}_{int(best_record["score"]*10000):04d}.pth')

        return model_file

    @staticmethod
    def load_model(config_file, saved_model_file):

        print(f"init model from: {config_file}")
        print(f"load model parameter from: {saved_model_file}")

        with open(config_file, 'r') as f:
            config_custom = json.load(f)

        #setup model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        if config_custom["model"]["transform_type"] in ["Rigid", "Similarity", "Affine"]:
            model = networks.sgmanet_affine(config_custom["model"]["param"], 
                                    config_custom["data"]["param"]["image_shape"],
                                    config_custom["model"]["transform_type"]).to(device)
        else:
            model = networks.sgmanet(config_custom["model"]["param"], 
                                    config_custom["data"]["param"]["image_shape"], 
                                    config_custom["model"]["int_steps"],
                                    config_custom["model"]["int_downsize"]).to(device)
        
        model_state = torch.load(saved_model_file, map_location=device)
        if "model_state_dict" in model_state:
            model.load_state_dict(model_state["model_state_dict"])
        else:
            model.load_state_dict(model_state)
        model.eval()

        return model, device

    def registration(self, moving, fix):

        moving_image = moving[0].unsqueeze(0).unsqueeze(0).to(self.device)
        fix_image = fix[0].unsqueeze(0).unsqueeze(0).to(self.device)

        moving_label = moving[2].unsqueeze(0).to(self.device)
        fix_label = fix[2].unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad(): 
            result_model = self.model(moving_image, fix_image, moving_label, fix_label, need_moved_label=True, need_deformation_filed=True)

        end_time = time.perf_counter()

        result = {}
        result["moved_image"] = result_model["moved_image"].cpu()
        result["moved_label"] = result_model["moved_label"].cpu()
        result["affine_matrix"] = result_model["matrix"].cpu() if "matrix" in result_model else None
        result["svf"] = result_model["displacement_filed"].cpu() if "displacement_filed" in result_model else None
        result["deformation"] = result_model["deformation_filed"].cpu() if "deformation_filed" in result_model else None
        result["run_time"] = end_time - start_time

        return result
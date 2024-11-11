from convexAdam.convex_adam_MIND import convex_adam_pt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import layers
import torch
import time

class convexadam_reg():
    def __init__(self, config) -> None:
       self.transform = layers.SpatialTransformerDouble(config["dataset_param"]["image_shape"])
    
    def registration(self, moving, fix):

        moving_image_numpy = moving[0].squeeze().numpy()   
        fix_image_numpy = fix[0].squeeze().numpy()

        start_time = time.perf_counter()

        displacements = convex_adam_pt(
            img_fixed=fix_image_numpy,
            img_moving=moving_image_numpy,
        )

        disp = torch.from_numpy(displacements).float()
        disp = disp.permute(3, 0, 1, 2).unsqueeze(0)

        moving_image = moving[0].unsqueeze(0).unsqueeze(0)
        moving_label_onehot = moving[2].unsqueeze(0)
        moved_image, moved_label = self.transform(moving_image, moving_label_onehot, disp, need_moved_label=True)

        end_time = time.perf_counter()

        result = {}
        result["moved_image"] = moved_image
        result["moved_label"] = moved_label
        result["affine_matrix"] = None
        result["svf"] = disp
        result["deformation"] = self.transform.grid + disp
        result["run_time"] = end_time - start_time
        return result
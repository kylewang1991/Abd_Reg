import ants
import time

class ants_reg():
    def __init__(self, config):
        if not (config["dataset"] in ["chaos", "amos"]):
            raise ValueError(f'dataset not supported! dataset={config["dataset"]}' )

        self.transform_type = "Rigid" if config["dataset"] == "chaos" else "Similarity"
    
    def registration(self, moving, fix):

        moving_path = moving[5]
        fix_path = fix[5]

        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]
        fix_image_path = fix_path["image"]
        fix_label_path = fix_path["label"]

        moving_image = ants.image_read(moving_image_path)
        fixed_image = ants.image_read(fix_image_path)

        start_time = time.perf_counter()

        result_ants = ants.registration(fixed_image, moving_image, 
                                type_of_transform=self.transform_type,
                                aff_metric="mattes",
                                verbose=False)
        
        end_time = time.perf_counter()
        
        moving_label = ants.image_read(moving_label_path)
        fix_label = ants.image_read(fix_label_path)

        moved_label = ants.apply_transforms(fix_label, moving_label, 
                                                transformlist= result_ants['fwdtransforms'],
                                                interpolator = 'genericLabel')



        result = {}
        result["moved_image"] = result_ants['warpedmovout']
        result["moved_label"] = moved_label
        result["affine_matrix"] = result_ants['fwdtransforms'][0]
        result["svf"] = None
        result["deformation"] = None
        result["run_time"] = end_time - start_time

        return result
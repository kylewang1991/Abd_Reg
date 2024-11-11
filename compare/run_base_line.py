class baseline_reg():
    def __init__(self, config) -> None:
        pass

    def registration(self, moving, fix):
        result = {}
        result["moved_image"] = moving[0]
        result["moved_label"] = moving[2]
        result["affine_matrix"] = None
        result["svf"] = None
        result["deformation"] = None

        return result
# Enhanced Multi-modal Abdominal Image Registration via Structural Awareness and Region-Specific Optimization

## Overview

This project contains the code for our paper, *"Enhanced Multi-modal Abdominal Image Registration via Structural Awareness and Region-Specific Optimization."* We address challenges in abdominal image registration by introducing the Modality Independent Structural Awareness (MISA) loss, which enhances accuracy through structural consistency and region-specific weighting. Our method outperforms state-of-the-art techniques, especially in capturing complex anatomical structures and maintaining tissue consistency across modalities, providing a valuable tool for multi-modal clinical imaging analysis.

## Code Structure

- **./compare**: Code for comparative experiments and model testing.
- **./dataset**: Dataset preprocessing scripts.
- **./SGMANet**: Core model implementation, including loss functions, evaluation metrics, and training scripts.

## Data Preparation

1. Download the dataset from [CHAOS](https://chaos.grand-challenge.org/Data/) and [AMOS22](https://amos22.grand-challenge.org/) websites.
2. Unzip the datasets and save them to `../CHAOS_Train_Sets` and `../amos22` folders.
3. Run the preprocessing scripts in the `./dataset` folder. If you save the data in different folders, adjust the data paths in these scripts and `.json` files accordingly.

## How to Reproduce the Results

### 1. Download Pre-trained Models

Download the pre-trained models from [this link](https://drive.google.com/drive/folders/1YWk_zeR6a7MxNa6TVor-6Thbwm_AVW7W?usp=sharing), and place them in the `./SGMANet/modules` folder.

### 2. Download Pre-generated Affine Transformation Matrices

To ensure consistency with the results reported in the paper, we have uploaded pre-generated affine transformation matrices. Each dataset has two corresponding folders: `affine_mi` and `affine_struct`. The download links and their storage paths are provided below:

- `../CHAOS_Train_Sets/affine_mi`: [Download Link](https://drive.google.com/drive/folders/1tkMFrkU5ycRhjY95bFK5VSRB4mSjsX5-?usp=sharing)
- `../CHAOS_Train_Sets/affine_struct`: [Download Link](https://drive.google.com/drive/folders/1tkMFrkU5ycRhjY95bFK5VSRB4mSjsX5-?usp=sharing)
- `../amos22/affine_mi`: [Download Link](https://drive.google.com/drive/folders/1tkMFrkU5ycRhjY95bFK5VSRB4mSjsX5-?usp=sharing)
- `../amos22/affine_struct`: [Download Link](https://drive.google.com/drive/folders/1tkMFrkU5ycRhjY95bFK5VSRB4mSjsX5-?usp=sharing)

### 3. Run the Tests

Use the `test_runner.py` to run the tests. See the following usage:

```
usage: test_runner.py [-h] [-m {method_options}] [-d {dataset_options}] [-v] [-t DATA_TYPE] [-p MODEL_PATH] [-n MAT_PATH] [-i INDEX]

options:
  -m, --method       method to be used
  -d, --data         dataset to use
  -v, --visual       save the moved image and label for visualization
  -t, --data_type    data type: train, valid, test
  -p, --model_path   path to the saved model
  -n, --mat          path to the linear transformation matrix
  -i, --index        index of the data to be tested
```

Due to version differences, some method names in the code do not match those in the paper. Below are the commands needed to reproduce the intra-patient experiment results:

- Baseline: `python ./compare/test_runner.py -m baseline -d chaos`
- AntsPy Syn: `python ./compare/test_runner.py -m sync -d chaos`
- NiftyReg: `python ./compare/test_runner.py -m niftyreg -d chaos`
- ConvexAdam: `python ./compare/test_runner.py -m convex_adam -d chaos`
- Voxelmorph: `python ./compare/test_runner.py -m voxelmorph -d chaos_affine_mi`
- Transmorph: `python ./compare/test_runner.py -m transmorph -d chaos_affine_mi`
- ConvNet (Ours): `python ./compare/test_runner.py -m sgmaconv -d chaos_affine_struct`
- TransNet (Ours): `python ./compare/test_runner.py -m sgmatrans_2 -d chaos_affine_struct`

For inter-patient experiments, change the dataset name from `chaos, chaos_affine_mi, chaos_affine_struct` to `amos, amos_affine_mi, amos_affine_struct`.

## How to Train

Use the `./SGMANet/run_train.py` script to start training. Example usage:

```bash
usage: run_train.py [-h] [-c CONFIG_PATH] [-d {dataset_options}] [-l LOAD]

options:
  -c, --config      path to the config JSON file
  -d, --data        dataset to use
  -l, --load        path to load the model
```

Example command to train TransNet (Ours) on the CHAOS dataset:

```bash
python ./SGMANet/run_train.py -c SGMANet/config/chaos_sgmatrans.json -d chaos_affine_struct
```

During training, all hyperparameter settings are saved in `./script/xxx.json` files, named in the format `[dataset]_[method].json`. Please refer to these files for specific configurations.

## License

This project is licensed under the MIT License. For more details, refer to the LICENSE file.

## Contact

For questions or suggestions, please contact [wzwt1991@163.com](mailto:wzwt1991@163.com).

## Third-Party Code

This project includes code from the following open-source projects:

- [Voxelmorph](https://github.com/voxelmorph/voxelmorph) - Licensed under the Apache License 2.0
- [Transmorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) - Licensed under the MIT License
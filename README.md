# PASIF: A plug-and-play framework enables tailored molecular design in zero-shot biological regimes
![图片加载失败](./picture/model-readme.svg)

The PASIF code is currently being organized.

## Installation
Run the following code to set up the environment:

    conda env create -f environment.yaml


## Weights and Datasets
Download the model weights via [Zendo](https://zenodo.org/records/20685422) and place them in the `./logs` dir. The expected file structure is shown below:

    -PASIF
        -logs
            -admet
            -affinity
            ...

The molecular data contained in the test set is sourced from [CBGBench](https://github.com/Edapinenut/CBGBench). Annotated data in the electron density task is available via [Zendo](https://zenodo.org/records/20625169). The dataset for training and evaluating the likelihood predictors is available via [Zendo](https://zenodo.org/records/20626350), which is sourced from [Deep-PK](https://biosig.lab.uq.edu.au/deeppk/) and [KGDiff](https://github.com/CMACH508/KGDiff).

All test data is available on [Zenodo](https://zenodo.org/records/20702136). Please download and place the files into the `./data` directory. The final directory structure should look like this:

    -PASIF
        -data
            -crossdocked_test
            -electron
            -pl
            ...

## Generation on Various Tasks

### Run Generation on the test examples
Please download the `case.zip` archive from [Zenodo](https://zenodo.org/records/20702136) and extract its contents. The resulting file path should be:

    -PASIF
        -case
            -admet
            -charge
            -leadopt
            -specificity

#### Target Selectivity
You can generate molecules on the test examples, with the following:

    python demo_specificity.py --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint} --classifier {affinity predictor checkpoint}

Since the atom mapping schemes vary across different Diffusion models, we trained corresponding affinity predictors for each scheme. The mapping between the Diffusion models and predictor weights is shown in the table below.

| Diffusion Model | Predictor Checkpoint |
| --------------- | ---------- |
| DiffBP          | ./logs/affinity/add_aromatic/self-train/checkpoints/180000.pt|
| TargetDiff      | ./logs/affinity/add_aromatic/self-train/checkpoints/180000.pt|
| DiffGui         | ./logs/affinity/diffgui/self-train/checkpoints/182000.pt|

For example, to implement this using DiffBP, execute the code below:

    python ./demo/demo_specificity.py --model_name diffbp --checkpoint ./logs/denovo/diffbp/pretrain/checkpoints/pretrained.pt --classifier ./logs/affinity/add_aromatic/self-train/checkpoints/180000.pt

#### ADMET Properties
You can draw samples with a specific favorable ADMET property for the test examples, with the following:

    python ./demo/demo_admet.py --prop {admet property name} --target {pocket file path}

#### Electron Density
For the global density constraints, you can running this:

    python ./demo/demo_charge_global.py --density_path {electron density file path} --target {pocket file path} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}

For the local density constraints, you can running this:

    python ./demo/demo_charge_local.py --density_path {electron density file path} --target {pocket file path} --mask {query point mask for key interaction regions} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}

#### Lead Optimization

You can perform lead optimization using molecular fragments by running this:

    python ./demo/demo_leadopt.py --frag {molecular frag file path} --target {pocket file path} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}


### Run Evaluation on the test set
Please ensure you have downloaded the test dataset and model weights as described above.

#### Target Selectivity

You can run evaluation with the following:
    
    python ./experiment/specificity.py --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint} --classifier {affinity predictor checkpoint}

If you opt for the DiffGui implementation, please execute the `specificity_diffgui.py` script.

#### ADMET Properties

Evaluation of various ADMET properties on the test set can be performed using the following command:

    python ./experiment/admet.py --prop {admet property name}

#### Electron Density

For the global electron density constraints, you can run evaluation as following:

    python ./experiment/eval_charge.py

For the local electron density constraints, you can run evaluation as following:

    python ./experiment/eval_charge_local.py

#### Lead Optimization

You can evaluate various pre-trained diffusion models on different subtasks by changing the config files:

    python ./experiment/leadopt.py --config {config file path} --out_root {output dir}

The mapping between the subtasks and the config folders is shown in the table below:

| SubTasks | Config Dir |
| -------- | ---------- |
| linker design | ./configs/linker/test |
| fragment grow | ./configs/frag/test |
| sidechain decoration | ./configs/sidechain/test |
| scaffold hopping | ./configs/scaffold/test |

Each directory contains the config files for different pre-trained diffusion models. If you opt for the DiffGui implementation, please download [Zendo](https://zenodo.org/records/20703901) and execute the `lead_opt.py` script.

## Training
To train a custom ADMET property predictor yourself, run the following command:

    python train_admet.py --prop [property name] --config [config file path] --predictor [predictor type: regression(reg)/classification(cls)]

You can also train your own affinity predictor by running the following code:

    python train_affinity.py --config [config file path]

You can train the diffusion generative model from scratch by executing the code below:

    python train.py --config [config file path]
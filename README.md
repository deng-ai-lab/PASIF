# PASIF: A plug-and-play framework enables tailored molecular design in zero-shot biological regimes
![图片加载失败](./picture/model-readme.svg)

The PASIF code is currently being organized.

## To Do List
The PASIF code is currently being organized.

1. Complete the installation section
2. Complete the run evaluation on the test set section

## Installation


## Weights and Datasets
Download the model weights via [Zendo](https://zenodo.org/records/20685422) and place them in the `./logs` dir. The expected file structure is shown below:

    -PASIF
        -logs
            -admet
            -affinity
            ...

The molecular data contained in the test set is sourced from [CBGBench](https://github.com/Edapinenut/CBGBench). Annotated data in the electron density task is available via [Zendo](https://zenodo.org/records/20625169). The dataset for training and evaluating the likelihood predictors is available via [Zendo](https://zenodo.org/records/20626350), which is sourced from [Deep-PK](https://biosig.lab.uq.edu.au/deeppk/) and [KGDiff](https://github.com/CMACH508/KGDiff).

All test data is available on [Zenodo](https://zenodo.org/records/20701480). Please download and place the files into the `./data` directory. The final directory structure should look like this:

    -PASIF
        -data
            -crossdocked_test
            -electron
            -pl
            ...

## Generation on Various Tasks

### Run Generation on the test examples
Please download the `case.zip` archive from Zenodo and extract its contents. The resulting file path should be:

    -PASIF
        -case
            -admet
            -charge
            -leadopt
            -specificity

#### target selectivity
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

#### admet properties
You can draw samples with a specific favorable ADMET property for the test examples, with the following:

    python ./demo/demo_admet.py --prop {admet property name} --target {pocket file path}

#### electron density
For the global density constraints, you can running this:

    python ./demo/demo_charge_global.py --density_path {electron density file path} --target {pocket file path} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}

For the local density constraints, you can running this:

    python ./demo/demo_charge_local.py --density_path {electron density file path} --target {pocket file path} --mask {query point mask for key interaction regions} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}

#### lead optimization

You can perform lead optimization using molecular fragments by running this:

    python ./demo/demo_leadopt.py --frag {molecular frag file path} --target {pocket file path} --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint}

### Run Evaluation on the test set
Please ensure you have downloaded the test dataset and model weights as described above.
#### target selectivity
You can run evaluation with the following:
    
    python specificity.py --model_name {diffusion base model name} --checkpoint {diffusion base model checkpoint} --classifier {affinity predictor checkpoint}

If you opt for the DiffGui implementation, please execute the `specificity_diffgui.py` script:

#### ADMET properties

Evaluation of various ADMET properties on the test set can be performed using the following command:

    python admet.py --prop {admet property name}

#### electron density



#### lead optimization


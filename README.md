
# A Benchmark for Efficient and Compositional Visual Reasoning

This reposity details the Compositional Visual Relations (CVR) benchmark. 

## Release Notes

* 09/07/2021: Paper submission to the Neurips 2022 Datasets and Benchmarks Track

## Citation

```stex
@article{zerroug2022benchmark,
  title={A Benchmark for Compositional Visual Reasoning},
  author={Zerroug, Aimen and Vaishnav, Mohit and Colin, Julien and Musslick, Sebastian and Serre, Thomas},
  journal={arXiv preprint arXiv:2206.05379},
  year={2022}
}
```

## Dataset

CVR evaluates visual reasoning model using 103 unique tasks. The code for automatically generating samples and the descriptions of each task are provided in `data_generation/tasks.py`. The generalization test set is generated using functions provided in `data_generation/generalization_tasks.py`.

Each sample of the dataset is an odd-one-out problem. The outlier is chosen among 4 images. The 4 images of each problem are provided in a single png file concatenated horizontally. The last image of the 4 is the outlier. The task index is provided with each problem.

## Dataset Generation

The dataset can be generated with the `generate_dataset.py` script

`python generate_dataset.py --data_dir <path to dataset> --task_idx a --seed 0 --train_size 10000 --val_size 500 --test_size 1000 --test_gen_size 1000 --image_size 128`

`task_idx` can be changed with an integer between 0 and 102 to generate a specific task. The dataset is generated with a fixed seed.

# Experiments

Currently, 5 models are evaluated on CVR; ResNet-50, ViT-small, SCL, WReN and SCL-ResNet-18. This benchmark evaluates Sample efficiency and compositionality. 

Each model is trained in several settings:
- on individual tasks (`ind`) or jointly on all tasks (`all`)
- using a random initialization or pretrained using self-supervision.
- on several data regimes; `20`, `50`, `100`, `200`, `500` and `1000` samples per task.

To train models in these settings, modify and run the job script `jobs/train_array.sh`. Choose `condition='ind'` and `cfg_list=(0 1 2 3 4 5)`.

Sample efficiency is measured from the accuracy on 6 data regimes `20`, `50`, `100`, `200`, `500` and `1000` samples. The *Area Under the Curve* (**AUC**) and *Sample Efficiency Score* (**SES**) scores are computed as follows:

```import numpy as np

def SES(accuracy):
  n_samples = np.array([20,50,100,200,500,1000])
  return np.sum(accuracy/(1+np.log(n_samples)))/(1/(1+np.log(n_samples))).sum()

def AUC(accuracy):
  return np.mean(accuracy)

```

Compositionality is evaluated in 2 conditions: the **curriculum condition** and the **reverse curriculum condition**. In the curriculum condition, models are finetuned on composition tasks after pretraining on their elementary components. In the reverse curriculum condition, models are evaluated on the elementary components after training on the compoitions.

To train and test models on the curriculum condition in the individual task training setting, modify and run `jobs/train_comp_ind.sh`. Choose `model_type` and `exp_index`.
To train and test models on the curriculum condition in the individual task training setting, modify and run `jobs/train_array.sh`.
Choose `condition='elem'` and `cfg_list=(7)` for pretraining then `condition='elem_comp'` and `cfg_list=(0 1 2 3 4 5)` for finetuning.

To train models on the reverse curriculum condition, modify and run the job script `jobs/train_array.sh`. Choose `condition='comp'` and `cfg_list=(7)` for the joint setting. Choose `condition='ind'` and `cfg_list=(7)` for the individual setting.

Evaluate models on the elementary tasks using `inference.py`.

## Self-Supervised Learning

To pretrain standard vision models on the dataset with self-supervised learning, we adapt the code provided by {moco-v3}[https://github.com/facebookresearch/moco-v3]. Training details are provided in the `ssl` folder.

## Results

The sample complexity results are reported for all models in all settings. Read the paper for detailed sample efficiency and compositionalty results.

<table>
    <thead>
        <tr>
            <th colspan=2>Setting</th>
            <th>Model</th>
            <th>AUC</th>
            <th>SES</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=10>Rand-Init</td> <td rowspan=5>Individual</td>
             <td>ResNet-50</td>     <td>33.7</td> <td>34.9</td> </tr>
        <tr> <td>ViT-small</td>     <td>31.3</td> <td>31.7</td> </tr>
        <tr> <td>SCl</td>           <td>29.9</td> <td>30.3</td> </tr>
        <tr> <td>WReN</td>          <td>33.4</td> <td>34.1</td> </tr>
        <tr> <td>SCL-ResNet-18</td> <td>38.4</td> <td>39.5</td> </tr>
        <tr>
            <td rowspan=5>Joint</td>
              <td>ResNet-50</td>    <td>36.0</td> <td>38.4</td> </tr>
        <tr> <td>ViT-small</td>     <td>28.4</td> <td>28.7</td> </tr>
        <tr> <td>SCL</td>           <td>32.2</td> <td>33.9</td> </tr>
        <tr> <td>WReN</td>          <td>30.9</td> <td>32.0</td> </tr>
        <tr> <td>SCL-ResNet-18</td> <td>37.6</td> <td>40.4</td> </tr>
        <tr>
            <td rowspan=4>SSL</td>
            <td rowspan=2>Individual</td>
            <td>ResNet-50</td>  <td>52.4</td> <td>54.5</td> </tr>
        <tr> <td>ViT-small</td> <td>54.9</td> <td>56.4</td> </tr>
        <tr>
            <td rowspan=2>Joint</td>
            <td>ResNet-50</td>  <td>57.0</td> <td>59.6</td> </tr>
        <tr> <td>ViT-small</td> <td>44.7</td> <td>46.3</td> </tr>        
    </tbody>
</table>
 
# Licence

CVR is released under the Apache License, Version 2.0.
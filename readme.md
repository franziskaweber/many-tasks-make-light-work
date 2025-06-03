# Many Tasks Make Light Work: Learning to Localise Medical Anomalies from Multiple Synthetic Tasks

This repository is a fork of the repository [Many tasks make light work: Learning to localise medical anomalies from multiple synthetic tasks](https://github.com/matt-baugh/many-tasks-make-light-workhttps://github.com/matt-baugh/many-tasks-make-light-work) which contains the implementation of the [paper of the same name](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_16) by M. Baugh et al..

I reproduced their experiments on the VinDr-CXR dataset but adapted the code such that it works on the publicly available kaggle version of this dataset.

## Environment Installation

Contrary to the original repository, I decided to set up the environment with conda. The environment can be created with: 

```
conda env create -f multitask_method_env.yml
```

and subsequently activated with:

```
conda activate multitask_method_env
```

## Data

In the paper, the [official version of the VinDr-CXR dataset from PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) is used. However, one can only download the dataset from this website as a credentialed user. Therefore, I decided to work with the [version of the dataset which is publicly available on kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data) instead.

As the kaggle version of the dataset does not include annotations or labels for the VinDr-CXR test set, I created my own training and test set from the training subset of the kaggle data. Similarly to the *Many Tasks Make Light Work* paper, I used 4000 healthy samples for training and 2000 samples, 1000 healthy and 1000 unhealthy ones, for testing. To recreate this training and test set, follow the instructions in `vindr_cxr_preparation.ipynb`. 

After having prepared the training and the test set, example images can be visualized with the code from the `Raw` section of `dataset_examples.ipynb`.

For the inter-dataset blending with the ImageNet dataset, I used the validation images from ILSVRC2012, instead of the training images like the  *Many Tasks Make Light Work* paper, as the validation set is smaller and therefore faster to download. The required files are `ILSVRC2012_devkit_t12.tar.gz` and `ILSVRC2012_img_val.tar` from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.

## Preprocessing

Before the images can be preprocessed, the paths in `multitask_method/paths.py` need to be set:

- `base_data_input_dir`: path to VinDr-CXR dataset which was prepared with `vindr_cxr_preparation.ipynb`
- `base_prediction_dir`: directory in which the predictions should be stored
- `base_log_dir`: directory in which the trained models should be stored

Additionally, the path to the `many-tasks-make-light-work` directory needs to be set in line 15 of `multitask_method/preprocessing/vindr_cxr_preproc.py` and line 10 of `multitask_method/preprocessing/imagenet_preproc.py`, the value of `base_data_input_dir` needs to be copied into `raw_root` in line 23 of `multitask_method/preprocessing/vindr_cxr_preproc.py` and the path to the ImageNet data needs to be stored in `raw_root` in line 13 and `imagenet_output_dir` in line 14 in `multitask_method/preprocessing/imagenet_preproc.py`.

After having set all these paths, the data can be preprocessed with:

```
python multitask_method/preprocessing/vindr_cxr_preproc.py
python multitask_method/preprocessing/imagenet_preproc.py
```

Subsequently, examples of preprocessed VinDr-CXR images can be visualized with the code from the `Preprocessed` section of `dataset_examples.ipynb`. 

## Training

To train the model on $T\in\{1, \dots, 4\}$ tasks, run

```
python train.py <path_to_repo>/experiments/exp_VINDR_low_res_<T>_train.py <F>
```

for $F\in\left\{0, \dots, \dbinom{5}{T} - 1\right\}$, i.e., for $F\in\{0, \dots, 4\}$ if $T\in\{1,4\}$ and $F\in\{0, \dots, 9\}$ if $T\in\{2,3\}$.

## Prediction

To get the predictions of the model trained on $T$ tasks, run:

```
python predict.py <path_to_repo>/experiments/exp_VINDR_low_res_<T>_train.py
```

To visualize exemplary predictions of this model, run: 

```
python show_predictions.py <path_to_repo>/experiments/exp_VINDR_low_res_<T>_train.py 'ensemble'
```

The exemplary predictions can then be found in `<base_prediction_dir>/exp_VINDR_low_res_<T>_train/ensemble/images`.

## Evaluation

To evaluate the predictions of the model trained on $T$ tasks, run:

```
python eval.py <path_to_repo>/experiments/exp_VINDR_low_res_<T>_train.py
```

The result of this evaluation is stored in `<base_prediction_dir>/exp_VINDR_low_res_<T>_train/results.json`.

## Comparison to *Many Tasks Make Light Work* Paper

I compared the results of the method on my manually created VinDr-CXR training and test set (column **Self-Created**) to the ones reported in the *Many Tasks Make Light Work* paper (columns **DDAD** and **VinDr**):

| Train/Val Split | DDAD (Sample-Wise) | VinDr (Sample-Wise) | Self-Created (Sample-Wise) |  DDAD (Pixel-Wise) | VinDr (Pixel-Wise) |  Self-Created (Pixel-Wise) |
|-----------------|------------------|--------------------|---------------------------|------------------|-------------------|---------------------------|
| 1/4             | 78.4             | 71.2               | 85.8                      | 21.1             | 21.4              | 24.8                      |
| 2/3             | **80.7**         | **73.8**           | **86.1**                  | 24.0             | **24.7**          | **26.2**                  |
| 3/2             | 80.4             | 73.3               | 85.8                      | **24.3**         | **24.7**          | 25.5                      |
| 4/1             | 80.5             | 73.6               | **86.1**                  | 23.5             | 24.5              | 25.5                      |

The results on my manually created test set are better than the ones on the DDAD and VINDr test set reported in the paper, potentially because we only considered images for which the 3 radiologists agreed on either *healthy* or *unhealthy*. Such images are probably easier to classify as *healthy* or *unhealthy* as images on which the radiologists did not agree on.
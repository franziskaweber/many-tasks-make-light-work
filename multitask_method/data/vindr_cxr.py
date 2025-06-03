import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from multitask_method.data.dataset_tools import DatasetContainer, DatasetCoordinator
from multitask_method.preprocessing.vindr_cxr_preproc import TRAIN_IMAGE_LABELS, TEST_IMAGE_LABELS, TRAIN_ANNOTATIONS, \
    TEST_ANNOTATIONS, TRAIN, TEST, generate_vindr_mask


class VinDrCXRDatasetContainer(DatasetContainer):
    def __init__(self, sample_ids: List[str], image_folder: Path, annotations_df: Optional[pd.DataFrame]):
        self.image_folder = image_folder
        self.annotations_df = annotations_df
        super().__init__(sample_ids, True)

    def load_sample(self, sample_id: str) -> Tuple[npt.NDArray[float], Optional[npt.NDArray[bool]]]:
        # noinspection PyTypeChecker
        sample_arr = np.load(self.image_folder / f'{sample_id}.npy')
        assert len(sample_arr.shape) == 2, f'Expected 2D image, got shape {sample_arr.shape}'

        if self.annotations_df is not None:
            sample_annotations = self.annotations_df[self.annotations_df['image_id'] == sample_id]
            sample_mask = generate_vindr_mask(sample_annotations, sample_arr)
        else:
            sample_mask = None

        return sample_arr[None], sample_mask


class VinDrCXRDatasetCoordinator(DatasetCoordinator):
    def __init__(self, dataset_root: Path, fullres: bool, ddad_split: bool, train: bool,
                 sample_limit: Optional[int] = None):
        # Datasets must be stored in a folder with the following structure:
        # dataset_root
        #   lowres
        #       annotations_test.csv
        #       annotations_train.csv
        #       image_labels_train.csv
        #       image_labels_test.csv
        #       train
        #           <sample_id>.npy
        #       test
        #           <sample_id>.npy
        #   fullres
        #       [same as lowres]

        self.dataset_root = dataset_root
        self.fullres = fullres
        self.train = train

        self.dset_folder = dataset_root / ('fullres' if fullres else 'lowres')

        self.image_labels = pd.read_csv(self.dset_folder / (TRAIN_IMAGE_LABELS if train else TEST_IMAGE_LABELS))
        self.annotations = pd.read_csv(self.dset_folder / (TRAIN_ANNOTATIONS if train else TEST_ANNOTATIONS))
        self.sample_folder = self.dset_folder / (TRAIN if train else TEST)

        if train:
            image_labels_sum = self.image_labels.groupby('image_id')['No finding'].sum()
            self.sample_ids = sorted(image_labels_sum[image_labels_sum == 3].index.tolist())

            num_samples = len(self.sample_ids)
            assert num_samples == 4000, f'Unexpected number of healthy samples in training set: {num_samples}'
        else:
            self.sample_ids = sorted([f.stem for f in self.sample_folder.iterdir() if f.is_file()])
            num_samples = len(self.sample_ids)
            assert num_samples == 2000, f'Unexpected number of samples in test set: {num_samples}'

        if sample_limit is not None:
            self.sample_ids = self.sample_ids[:sample_limit]

    def make_container(self, sample_indices: List[int]) -> DatasetContainer:

        return VinDrCXRDatasetContainer([self.sample_ids[i] for i in sample_indices], self.sample_folder,
                                        None if self.train else self.annotations)

    def dataset_size(self):
        return len(self.sample_ids)

    def dataset_dimensions(self) -> int:
        return 2

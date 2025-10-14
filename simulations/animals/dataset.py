from torch.utils.data import Dataset
import h5py
import numpy as np

from typing import NamedTuple, Optional, Sequence, Union
import os

from sklearn.feature_selection import chi2, SelectKBest


class AnimalsConfig(NamedTuple):
    # Can reduce the number of features to use
    num_features: int = 512

    # Test on train, test (and val) features extraction
    chi2_test_on_feat_sel: bool = (False,)

    # Can either select all classes, or a subset, selected by class id or class label
    select_classes: Optional[Union[Sequence[int], Sequence[str]]] = None


class AnimalsDataset(Dataset):
    def __init__(
        self,
        config: AnimalsConfig,
        transform=None,
        target_transform=None,
        train: bool = True,
        val: bool = False,
        sel_feat: Optional[Union[Sequence[int], Sequence[str]]] = None,
        path: str = "animals/animals_features.h5",
    ):
        self.config = config
        self.selected_feat = sel_feat

        with h5py.File(
            os.path.join(os.environ["DATA"], path),
            "r",
        ) as dset:
            class_ids = dset.attrs["class_ids"]
            class_labels = dset.attrs["class_labels"]

            class_id2label = {k: v for k, v in zip(class_ids, class_labels)}

            def _class_selected(k):
                return (
                    self.config.select_classes is None
                    or (
                        isinstance(self.config.select_classes[0], int)
                        and int(k) in self.config.select_classes
                    )
                    or (
                        isinstance(self.config.select_classes[0], str)
                        and class_id2label[int(k)] in self.config.select_classes
                    )
                )

            split = "validation" if val else "train" if train else "test"
            data = {
                int(k): v[:, :] for k, v in dset[split].items() if _class_selected(k)
            }

            self.class_id2label = {
                k: v for k, v in zip(class_ids, class_labels) if _class_selected(k)
            }
            self.class_label2id = {
                v: k for k, v in zip(class_ids, class_labels) if _class_selected(k)
            }
            self.classes = list(self.class_label2id.keys())

        self.transform = transform
        self.target_transform = target_transform

        self.xs = np.concatenate([data[k] for k in sorted(data.keys())])
        self.ys = np.concatenate([[k] * len(data[k]) for k in sorted(data.keys())])

        # Select top num_features using chi-squared test
        if self.config.num_features < self.xs.shape[1]:
            if self.config.chi2_test_on_feat_sel is True:
                selector = SelectKBest(score_func=chi2, k=self.config.num_features)
                self.xs = selector.fit_transform(self.xs, self.ys)
                self.selected_feat = selector.get_support(indices=True)
            else:
                if self.selected_feat is not None:
                    self.xs = self.xs[:, self.selected_feat]
                else:
                    self.xs = self.xs[:, : self.config.num_features]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x, y = self.xs[idx], self.ys[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def get_selected_feat(self):
        return self.selected_feat

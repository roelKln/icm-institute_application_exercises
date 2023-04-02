import tarfile
import gdown
import torchio as tio
from pathlib import Path
import numpy as np
from torch.utils.data import random_split, DataLoader, RandomSampler

#class MedicalDecathlonDataModule:
#    def __init__(self):
#        super().__init__()
#        pass


class MedicalDecathlonDataModule:
    '''
    Dataset class for the Medical Segmentation Decathlon Task04_Hippocampus
    http://medicaldecathlon.com/

    Based on the torch.ai tutorial, but without using pytorch-lightning:
    https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb#scrollTo=KuhTaRl3vf37
    '''
    def __init__(self, task, google_id, batch_size, train_val_ratio):
        super().__init__()
        self.task = task
        self.google_id = google_id
        self.batch_size = batch_size
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio        
        # empty defaults, to be assigned
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.prepare_data()
        self.setup()

    def get_max_shape(self, subjects):
        ''' Over N subjects, finds the maximum size in x, y and z directions.'''
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def download_data(self):
        ''' Makes sure data is downloaded and unpacked '''
        tarfile_path = self.dataset_dir.with_suffix('.tar')
        if self.dataset_dir.is_dir():
            # means we have already downloaded and unpacked the data
            pass
        elif tarfile_path.is_file():
            # means we have already downloaded, but we still need to unpack
            with tarfile.open(str(tarfile_path)) as f:
                f.extractall(str(tarfile_path.parent))
        else:
            # means we need to download and unpack     
            # NOTE: automatic downloading might not be possible if the dataset
            # is being downloaded by a lot of people, in this case you have to
            # download it manually
            url = f'https://drive.google.com/uc?id={self.google_id}'
            gdown.download(url, str(tarfile_path), quiet=False)

            with tarfile.open(str(tarfile_path)) as tarf:
                tarf.extractall(str(tarfile_path.parent))

        def _get_niis(splitdir : Path):
            return sorted(p for p in splitdir.glob('*.nii.*') if not p.name.startswith('.'))

        image_training_paths = _get_niis(self.dataset_dir / 'imagesTr')
        label_training_paths = _get_niis(self.dataset_dir / 'labelsTr')
        image_test_paths = _get_niis(self.dataset_dir / 'imagesTs')
        return image_training_paths, label_training_paths, image_test_paths    

    def prepare_data(self):
        ''' Set subjects '''
        image_training_paths, label_training_paths, image_test_paths = self.download_data()

        self.subjects = []
        assert len(image_training_paths) == len(label_training_paths), "Not all training images have a label"
        for image_path, label_path in zip(image_training_paths, label_training_paths):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)
        
        self.test_subjects = []
        for image_path in image_test_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            self.test_subjects.append(subject)
    
    def get_preprocessing_transform(self):
        ''' Get image preprocessor: norm -> CropOrPad -> OneHot '''
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
            tio.EnsureShapeMultiple(8), # for U-Net downsampling
            tio.OneHot(),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        ''' 
        Get data augmenter:
        affine -> gamma noise -> gaussian noise -> motion artifact -> bias field artifact 
        '''
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25)
        ])
        return augment
    
    def setup(self):
        ''' Create train, val and test datasets '''
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self, steps_per_epoch=None, num_workers=1):
        ''' Returns a dataloader for the train set '''
        if steps_per_epoch is None:
            return DataLoader(self.train_set, self.batch_size, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=num_workers)
        else:
            sampler = RandomSampler(self.train_set, replacement=True, num_samples=(steps_per_epoch * self.batch_size))
            return DataLoader(self.train_set, self.batch_size, sampler=sampler,
                            pin_memory=True, num_workers=num_workers)

    def val_dataloader(self, num_workers=1):
        ''' Returns a dataloader for the validation set '''
        return DataLoader(self.val_set, self.batch_size, shuffle=False,
                        drop_last=False, pin_memory=True, num_workers=num_workers)

    def test_dataloader(self, num_workers=1):
        ''' Returns a dataloader for the test set '''
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                        drop_last=False, pin_memory=True, num_workers=num_workers)

if __name__ == "__main__":
    print("Testing dataset init...")
    data = MedicalDecathlonDataModule(task='Task04_Hippocampus', 
                                    google_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
                                    batch_size=16,
                                    train_val_ratio=0.8)
    print("Succesfull init")
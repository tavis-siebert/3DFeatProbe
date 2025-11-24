from torch.utils.data import DataLoader
from mapanything.datasets import get_test_data_loader, get_train_data_loader, get_test_many_ar_data_loader

def build_wai_dataloader(
    dataset: str, 
    num_workers: int, 
    test: bool, 
    multi_res: bool=False,
    batch_size: int=None, 
    max_num_of_imgs_per_gpu: int=None
) -> DataLoader:
    """
    Builds data loaders for training or testing.

    Args:
        dataset (str): Dataset specification string.
        num_workers (int): Number of worker processes for data loading.
        test (bool): Boolean flag indicating whether this is a test dataset.
        multi_res (bool): Whether to use multi-resolution data loadin for test datasets. Defaults to False
        batch_size (int): Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu (int): Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        if multi_res:
            loader = get_test_many_ar_data_loader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_mem=True,
                drop_last=False,
            )
        else:
            loader = get_test_data_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_mem=True,
                shuffle=False,
                drop_last=False,
            )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    return loader

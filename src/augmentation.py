from torchvision import transforms

# TODO: finalize the augmentation pipeline
# TODO: parameters should be configurable
def get_augmentation():
    """
    Get the augmentation pipeline for the data
    
    Returns:
    torchvision.transforms.Compose: augmentation pipeline
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomAffine(0, shear=15),
        transforms.RandomAffine(0, scale=(0.8, 1.2)),

    ])
from typing import List

from torch.utils.data import DataLoader

# Dataset Interface to act as dataset object for opinion triplet extraction system. 
# If you want to add your own dataset, feel free to create new Dataset Instance...
class IDataset:
    def get_stats(self) -> List[str]:
        pass 

    def get_sents(self) -> List[str]:
        pass


# Loader Interface to load and batch data from datasets into data loader (pytorch)
class ILoader:
    def get_train_dataset(self) -> 'IDataset' :
        pass 

    def get_val_dataset(self) -> 'IDataset':
        pass 

    def get_test_dataset(self) -> 'IDataset':
        pass 

    def get_train_loader(self) -> DataLoader:
        pass 

    def get_val_loader(self) -> DataLoader:
        pass 

    def get_test_loader(self) -> DataLoader:
        pass


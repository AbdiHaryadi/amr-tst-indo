from torch.utils.data import DataLoader
from src.loader.interface import ILoader

from src.loader.parser import parse
from src.loader.dataset import HotelDataset

from src.constant import Path, DatasetType

class HotelLoader(ILoader):
    def __init__(self, tokenizer, configs):
        self.configs = configs
        self.tokenizer = tokenizer
        self.is_loaded = False

        self.__load()
        self.is_loaded = True

    def get_train_dataset(self):
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.train_dataset
    
    def get_test_dataset(self) -> DataLoader:
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.test_dataset

    def get_val_dataset(self) -> DataLoader:
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.val_dataset

    def get_train_loader(self) -> DataLoader:
        dataset = self.get_train_dataset()
        batch_size = self.configs.get("trainer").get("batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
    
    def get_test_loader(self):
        dataset = self.get_test_dataset()
        batch_size = self.configs.get("trainer").get("eval_batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

    def get_val_loader(self):
        dataset = self.get_val_dataset()
        batch_size = self.configs.get("trainer").get("eval_batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

    def __load(self):
        if self.configs is None:
            raise ValueError("config not initialized")

        mode = self.configs.get("loader").get("mode")
        print(mode)
        if mode == DatasetType.FILTERED:
            train_path = Path.TRAIN_FILTERED_PATH
        elif mode == DatasetType.ANNOTATION:
            train_path = Path.TRAIN_ANNOTATION_PATH
        else:
            train_path = Path.TRAIN_UNFILTERED_PATH
            
        test_path = Path.TEST_UNFILTERED_PATH
        val_path = Path.VAL_UNFILTERED_PATH

        if mode == DatasetType.ANNOTATION:
            test_path = Path.TEST_ANNOTATION_PATH
            val_path = Path.VAL_ANNOTATION_PATH

        seperator = self.configs.get("loader").get("seperator")
        with open(train_path, "r", encoding="UTF-8") as train_file:
            train_sents, train_labels = parse(train_file, seperator)
        with open(test_path, "r", encoding="UTF-8") as test_file:
            test_sents, test_labels = parse(test_file, seperator)
        with open(val_path, "r", encoding="UTF-8") as val_file:
            val_sents, val_labels = parse(val_file, seperator)

        train_params = {
            "sents": train_sents,
            "labels": train_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.train_dataset = HotelDataset(**train_params)

        val_params = {
            "sents": val_sents,
            "labels": val_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.val_dataset = HotelDataset(**val_params)

        test_params = {
            "sents": test_sents,
            "labels": test_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.test_dataset = HotelDataset(**test_params)

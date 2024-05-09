from typing import List
from transformers.generation_utils import GenerationMixin

# Trainer Interface for fine-tuning pretrain model, can be used for other pre-trained model types...

class ITrainer:
    def fit(self):
        pass

    def save(self):
        pass

    def load(self, path: str):
        pass

    def get_model(self) -> GenerationMixin:
        pass

    def training_report(self) -> List[str]:
        pass

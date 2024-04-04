from typing import List 

# Generator Interface to predict the triplet opinion from review text using generative approach.
# Feel free to adjust the interface based on pretrained model... 
class IGenerator:
    def generate(self, sents: List[str], implicit:bool=False, fix:bool=False) -> List[str]:
        pass 
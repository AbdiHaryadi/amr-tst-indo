import os
import yaml
import random 
import torch 
import numpy as np

from torch import cuda

from src.constant import Path



EMPTY = ''
def extract(sequence):
    extractions = []
    triplets = sequence.split("; ")
    for elem in triplets:
        elem = elem[1:-1] # Remove the in the start "("  and at the end ")".
        try:
            a, b, c = elem.split(', ')
        except ValueError:
            a, b, c = '', '', ''
        
        a = a.strip()
        b = b.strip()
        c = c.strip()
        # Postprocess...
        if (a == EMPTY or b == EMPTY or c == EMPTY) or (a,b,c) in extractions:
            continue
        extractions.append((a, b, c)) 
    return extractions

def get_config(path):
    return yaml.safe_load(open(path, "r"))


def get_latest_version(folder, prefix, mode="patch"):
    """
    mode --> (major | minor | patch)
    """
    files = os.listdir(folder)
    versions = []
    for file in files:
        if file.startswith(prefix):
            version = [int(ver) for ver in file.split("-")[-1].split(".")]
            versions.append(version)

    if not versions:
        return "0.0.0"

    major, minor, patch = sorted(versions, reverse=True)[0]
    if mode == "major":
        return f"{major+1}.0.0"
    elif mode == "minor":
        return f"{major}.{minor+1}.0"
    elif mode == "patch":
        return f"{major}.{minor}.{patch+1}"
    else:
        raise ValueError("only support (major | minor | patch)")


def json_to_text(json):
    msg = ""
    for key, val in json.items():
        msg += f"{key}: {val}\n"
    return msg

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device():
    device = "cpu"
    if cuda.is_available():
        device = torch.device("cuda") # First GPU
    return device

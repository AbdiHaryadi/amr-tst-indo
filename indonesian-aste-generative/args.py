import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="prefix for models")
    parser.add_argument("--config", required=True, help="config for models")
    parser.add_argument("--do-train", action='store_true', help="perform training")
    parser.add_argument("--do-test", action='store_true', help="perform testing")
    return parser.parse_args()




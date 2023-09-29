import yaml
import os
import sys

class TrainConfig():
    def __init__(self):
        if 'train.yaml' in os.listdir(os.getcwd()):
            self.config_file = os.path.join(os.getcwd(), 'train.yaml')
            with open(self.config_file, 'r') as file:
                self.params = yaml.safe_load(file)
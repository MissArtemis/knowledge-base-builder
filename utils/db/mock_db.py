import os

import pandas as pd

from utils.config import ConfigManager

class MockDB:
    def __init__(self):
        self.config = ConfigManager()
        self.query_data = pd.read_pickle(self.config.query_data_path)
        self.doc_data = pd.read_pickle(self.config.doc_data_path)
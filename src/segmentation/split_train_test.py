from dataclasses import dataclass
from urllib.request import DataHandler


@dataclass
class SplitTrainTest:
    data_handler: DataHandler

    def split(self):
        pass
from elephas.utils.rdd_utils import to_simple_rdd
from pyspark import SparkContext, SparkConf
from elephas.spark_model import SparkModel
from dataclasses import dataclass


@dataclass
class ElphasIntegration:
    model: any
    train_data: list
    rdd: any = None
    spark_model: any = None

    def __post_init__(self):
        self.__config()
    
    def __config(self):
        conf = SparkConf().setAppName('medusa_app').setMaster('local[8]')
        sc = SparkContext(conf=conf)
        self.rdd = to_simple_rdd(sc, self.train_data)
        self.spark_model = SparkModel(self.model, frequency='epoch', mode='asynchronous')
    
    def train(self, epochs: int, batch_size: int, validation_split: float):
        self.spark_model.fit(self.rdd, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split)
    
    def predict(self, test_data: list):
        return self.spark_model.predict(test_data)

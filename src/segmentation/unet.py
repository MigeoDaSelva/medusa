from dataclasses import dataclass
from tensorflow.keras import layers


@dataclass
class UNet:
    img_size: tuple
    num_classes: int
    model: any = None

    def __post_init__(self):
        self.__get_model()
    
    def __get_model(self):
        # Model creation here
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    def summary(self):
        self.model.summary()
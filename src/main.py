from src.data_handler import DataHandler
from src.elphas.elphas_integration import ElphasIntegration
from src.segmentation.split_train_test import SplitTrainTest
from src.segmentation.unet import UNet
from src.segmentation.vetorizer import VectorizerData


if __name__=="__main__":

    data_dir = "./dataset/pets"
    batch_size = 32
    num_classes = 3
    img_size = (160, 160)

    data_handler = DataHandler(data_dir)

    print("Split dataset...")
    split_dataset = SplitTrainTest(data_handler)
    x_train, y_train, x_test, y_test = split_dataset.split()

    print("Vectorize dataset...")
    train_data = VectorizerData(batch_size=batch_size, img_size=img_size, input_paths=x_train, mask_paths=y_train)
    test_data = VectorizerData(batch_size=batch_size, img_size=img_size, input_paths=x_test, mask_paths=y_test)

    print("Build the model...")
    unet = UNet(img_size=img_size, num_classes=num_classes)
    unet.summary()

    print("Training the model with elphas...")
    elphas_integration = ElphasIntegration(unet, train_data=train_data)
    elphas_integration.train(epochs=15, batch_size=batch_size, validation_split=0.1)

    print("Predict...")
    results = elphas_integration.predict(test_data=test_data)


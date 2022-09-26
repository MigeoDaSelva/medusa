from dataclasses import dataclass
import os


@dataclass
class DataHandler:
    data_dir: str 

    def get_input_images(self) -> list:
        input_path = self.data_dir + "/images"
        input_images = sorted(
            [
                os.path.join(input_path, file_name)
                for file_name in os.listdir(input_path)
                if file_name.endswith(".jpg")
            ]
        )

        return input_images
    
    def get_mask_images(self) -> list:
        mask_path = self.data_dir + "/mask"
        mask_images = sorted(
            [
                os.path.join(mask_path, file_name)
                for file_name in os.listdir(mask_path)
                if file_name.endswith(".jpg")
            ]
        )

        return mask_images

import yaml
import cv2

def load_config(config_path):
    """ Load configuration from a YAML file. """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_image(image_path):
    """ Load an image using OpenCV. """
    return cv2.imread(image_path)

def save_model(model, path):
    """ Save the trained model. """
    model.save(path)
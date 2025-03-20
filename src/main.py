from InputProcessor import InputProcessor
import cv2

# Instantiate the processor
data_dir = "data/"
processor = InputProcessor(data_dir)

processor.load_images("00")
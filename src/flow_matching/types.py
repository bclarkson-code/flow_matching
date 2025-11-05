from PIL import Image


DatasetElement = dict[str, Image.Image | dict[str, str]]

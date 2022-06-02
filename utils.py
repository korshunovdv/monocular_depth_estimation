import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def open_img(img_fname: str) -> np.array:
    image = Image.open(img_fname)
    return np.array(image)


def show_img(image: np.array) -> None:
    fig = plt.figure(figsize=(14, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def depth_to_heatmap(depht_arr: np.array, reverse: bool = True) -> np.array:
    colormap = plt.get_cmap('inferno')
    depht_arr = depht_arr / depht_arr.max()
    
    if reverse:
        mask = np.ma.masked_where(depht_arr==0, depht_arr)
        depht_arr = 1 - depht_arr
        depht_arr[mask.mask] = 0

    heatmap = colormap(depht_arr)
    return heatmap


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

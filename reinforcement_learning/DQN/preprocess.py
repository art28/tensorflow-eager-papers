import cv2
import numpy as np


def convert(obs):
    """ grayscale, resize, float-cast atari game observation image
    Args:
        obs: observation image
    Returns:
        processed grayscale-float image
    """
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray_cut = gray[32:193,8:152]  # cut the scoreboard, edge part
    gray_cut_float = gray_cut.astype(np.float32)/255.
    return gray_cut_float


def downsample(obs, fx, fy):
    """ resizing the observation image
    Args:
        obs: observation image
    Returns:
        downsampled image
    """
    downsampled = cv2.resize(obs, (fx, fy))
    return downsampled


def preprocess(obs, fx, fy):
    """ preprocessing wrapped for atrai game observation
    Args:
        obs: observation image
    Returns:
        downsampled - grayscale - float image
    """
    gray_cut_float = convert(obs)
    downsampled = downsample(gray_cut_float, fx, fy)
    return downsampled

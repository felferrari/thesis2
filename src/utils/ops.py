import json
import numpy as np
from osgeo import gdal_array
from osgeo import gdal, gdalconst
from pathlib import Path
from typing import Union
import yaml 
from itertools import chain, product

gdal.UseExceptions()

def remove_outliers(data, significance = 0.01):
    clip_values = np.percentile(data, (significance, 100-significance), axis=(0,1))
    data = np.clip(data, clip_values[0], clip_values[1])
    return data


def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        json.dump(dict_, f, indent=4)

def save_yaml(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        yaml.dump(dict_, f, default_flow_style=False)

def load_yaml(file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_opt_image(img_file) -> np.ndarray:
    """load optical data.

    Args:
        img_file (str): path to the geotiff optical file.

    Returns:
        array:numpy array of the image.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1) / 10000

def load_sb_image(img_file) -> np.ndarray:
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    return img

def load_ml_image(img_file) -> np.ndarray:
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_SAR_image(img_file) -> np.ndarray:
    """load SAR image, converting from Db to DN.

    Args:
        img_file (str): path to the SAR geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    #img = 10**(img/10) 
    img[np.isnan(img)] = 0
    return np.moveaxis(img, 0, -1)

def save_feature_map(path_to_file, tensor, index = None):
    if index is not None:
        fm = tensor[index]


def save_geotiff(base_image_path, dest_path, data, dtype, gdal_options = ['COMPRESS=DEFLATE']):
    """Save data array as geotiff.
    Args:
        base_image_path (str): Path to base geotiff image to recovery the projection parameters
        dest_path (str): Path to geotiff image
        data (array): Array to be used to generate the geotiff
        dtype (str): Data type of the destiny geotiff: If is 'byte' the data is uint8, if is 'float' the data is float32
    """
    base_image_path = str(base_image_path)
    base_data = gdal.Open(base_image_path, gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()
    dest_path = str(dest_path)
    
    # if len(gdal_options) == 0:
    #     if dtype == 'byte':
    #         gdal_options = ['COMPRESS=JPEG']
    #     elif dtype == 'float':
    #         gdal_options = ['COMPRESS=DEFLATE']
    
    if len(data.shape) == 2:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Byte, options = gdal_options)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Float32, options = gdal_options)
            data = data.astype(np.float32)
        elif dtype == 'uint16':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_UInt16, options = gdal_options)
            data = data.astype(np.uint16)
    elif len(data.shape) == 3:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Byte, options = gdal_options)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Float32, options = gdal_options)
            data = data.astype(np.float32)
        elif dtype == 'uint16':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_UInt16, options = gdal_options)
            data = data.astype(np.uint16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    if len(data.shape) == 2:
        target_ds.GetRasterBand(1).WriteArray(data)
    elif len(data.shape) == 3:
        for band_i in range(1, data.shape[-1]+1):
            target_ds.GetRasterBand(band_i).WriteArray(data[:,:,band_i-1])
    target_ds = None

def set_from_combinations(combinations):
    if len(combinations) == 1:
        return set(combinations[0])
    elif len(combinations) == 2:
        return set(chain(combinations[0], combinations[1]))
    
def possible_combinations(combinations):
    if len(combinations) == 1:
        return (tuple(combinations[0]),)
    elif len(combinations) == 2:
        comb = tuple(product(combinations[0], combinations[1]))
        if len(comb) == 0:
            return (tuple(),)
        else:
            return comb
from pathlib import Path
from src.utils.ops import load_opt_image, save_geotiff, remove_outliers, set_from_combinations
import numpy as np
from tqdm import tqdm
from osgeo import gdal, ogr, gdalconst
from skimage.morphology import disk, dilation, erosion, area_opening
from einops import rearrange
import pandas as pd

gdal.UseExceptions()

def generate_images_statistics(cfg_data, data_path, load_image, significance = 0):
    #general_stats = pd.DataFrame(columns=['cond', 'band', 'mean', 'std', 'max', 'min'])
    general_stats = None
    for k in cfg_data.condition:
        data_path = Path(data_path)
        stats = None
        pbar = tqdm(set_from_combinations(cfg_data.condition[k]), desc=f'Generating Statistics for {k} condition')
        for img_i in pbar:
            img_file = data_path / cfg_data.imgs[img_i]
            data = load_image(img_file)
            data = remove_outliers(data, significance)
            if stats is None:
                stats = {
                    'means': np.expand_dims(data.mean(axis = (0,1)), axis=0),
                    'stds': np.expand_dims(data.std(axis = (0,1)), axis=0),
                    'maxs': np.expand_dims(data.max(axis = (0,1)), axis=0),
                    'mins': np.expand_dims(data.min(axis = (0,1)), axis=0),
                }
            else:
                stats = {
                    'means': np.concatenate((stats['means'], np.expand_dims(data.mean(axis = (0,1)), axis=0)), axis= 0),
                    'stds': np.concatenate((stats['stds'], np.expand_dims(data.std(axis = (0,1)), axis=0)), axis= 0),
                    'maxs': np.concatenate((stats['maxs'], np.expand_dims(data.max(axis = (0,1)), axis=0)), axis= 0),
                    'mins': np.concatenate((stats['mins'], np.expand_dims(data.min(axis = (0,1)), axis=0)), axis= 0),
                }
        if stats is not None:
            stats = {
                'band': np.arange(stats['means'].shape[1]),
                'cond': [k] * stats['means'].shape[1],
                'mean': stats['means'].mean(axis=0),
                'std': stats['stds'].mean(axis=0),
                'max': stats['maxs'].max(axis=0),
                'min': stats['mins'].min(axis=0)
            }
            general_stats = pd.concat([general_stats, pd.DataFrame(stats)])
        
    return general_stats

def generate_tiles(cfg):
    base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]

    shape = load_opt_image(base_image).shape[0:2]

    tiles = np.zeros(shape, dtype=np.uint8).reshape((-1,1))
    idx_matrix = np.arange(shape[0]*shape[1], dtype=np.uint32).reshape(shape)

    tiles_idx = []
    for hor in np.array_split(idx_matrix, cfg.site.tiles_params['lines'], axis=0):
        for tile in np.array_split(hor, cfg.site.tiles_params['columns'], axis=1):
            tiles_idx.append(tile)

    
    for i, tile in enumerate(tiles_idx):
        if i in cfg.site.tiles_params['train_tiles']:
            tiles[tile] = 1

    tiles = tiles.reshape(shape)

    tiles_file = Path(cfg.path.tiles)
    save_geotiff(
        base_image, 
        tiles_file, 
        tiles, 
        'byte'
        )

def generate_labels(cfg):

    base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]

    f_yearly_def = Path(cfg.path.prodes.yearly_deforestation)
    v_yearly_def = ogr.Open(str(f_yearly_def))
    l_yearly_def = v_yearly_def.GetLayer()

    f_previous_def = Path(cfg.path.prodes.previous_deforestation)
    v_previous_def = ogr.Open(str(f_previous_def))
    l_previous_def = v_previous_def.GetLayer()

    f_no_forest = Path(cfg.path.prodes.no_forest)
    if f_no_forest.exists():
        v_no_forest = ogr.Open(str(f_no_forest))
        l_no_forest = v_no_forest.GetLayer()
    else:
        l_no_forest = None

    f_residual = Path(cfg.path.prodes.residual)
    if f_residual.exists():
        v_residual = ogr.Open(str(f_residual))
        l_residual = v_residual.GetLayer()
    else:
        l_residual = None


    f_hydrography = Path(cfg.path.prodes.hydrography)
    v_hydrography = ogr.Open(str(f_hydrography))
    l_hydrography = v_hydrography.GetLayer()


    base_data = gdal.Open(str(base_image), gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()

    #train label
    train_output = cfg.path.label.train

    target_train = gdal.GetDriverByName('GTiff').Create(train_output, x_res, y_res, 1, gdal.GDT_Byte)
    target_train.SetGeoTransform(geo_transform)
    target_train.SetSpatialRef(crs)
    target_train.SetProjection(proj)

    band = target_train.GetRasterBand(1)
    band.FlushCache()
    train_year = cfg.general.year.train
    where_past = f'"year"<={train_year -1}'
    where_ref = f'"year"={train_year}'

    gdal.RasterizeLayer(target_train, [1], l_previous_def, burn_values=[2])
    if l_no_forest is not None:
        gdal.RasterizeLayer(target_train, [1], l_no_forest, burn_values=[3])
    if l_residual is not None:
        gdal.RasterizeLayer(target_train, [1], l_residual, burn_values=[3])
        
    gdal.RasterizeLayer(target_train, [1], l_hydrography, burn_values=[3])



    l_yearly_def.SetAttributeFilter(where_past)
    gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[2])

    l_yearly_def.SetAttributeFilter(where_ref)
    gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[1])

    rasterized_data = target_train.ReadAsArray() 

    defor_data = rasterized_data == 1
    defor_data = defor_data.astype(np.uint8)

    def_inner_buffer = cfg.general.buffer.inner
    def_outer_buffer = cfg.general.buffer.outer
    border_data = dilation(defor_data, disk(def_inner_buffer)) - erosion(defor_data, disk(def_outer_buffer))

    rasterized_data[border_data==1] = 3

    defor_label = (rasterized_data==1).astype(np.uint8)
    defor_remove = defor_label - area_opening(defor_label, cfg.general.area_min)
    rasterized_data[defor_remove == 1] = 3

    target_train.GetRasterBand(1).WriteArray(rasterized_data)
    target_train = None

    #test label
    test_output = cfg.path.label.test

    target_test = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Byte)
    target_test.SetGeoTransform(geo_transform)
    target_test.SetSpatialRef(crs)
    target_test.SetProjection(proj)

    band = target_test.GetRasterBand(1)
    band.FlushCache()
    test_year =  cfg.general.year.test
    where_past = f'"year"<={test_year -1}'
    where_ref = f'"year"={test_year}'

    gdal.RasterizeLayer(target_test, [1], l_previous_def, burn_values=[2])
    if l_no_forest is not None:
        gdal.RasterizeLayer(target_test, [1], l_no_forest, burn_values=[3])
    gdal.RasterizeLayer(target_test, [1], l_hydrography, burn_values=[3])
    if l_residual is not None:
        gdal.RasterizeLayer(target_test, [1], l_residual, burn_values=[3])

    l_yearly_def.SetAttributeFilter(where_past)
    gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[2])

    l_yearly_def.SetAttributeFilter(where_ref)
    gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[1])

    rasterized_data = target_test.ReadAsArray() 

    defor_data = rasterized_data == 1
    defor_data = defor_data.astype(np.uint8)

    def_inner_buffer = cfg.general.buffer.inner
    def_outer_buffer = cfg.general.buffer.outer
    border_data = dilation(defor_data, disk(def_inner_buffer)) - erosion(defor_data, disk(def_outer_buffer))

    del defor_data

    rasterized_data[border_data==1] = 3

    defor_label = (rasterized_data==1).astype(np.uint8)
    defor_remove = defor_label - area_opening(defor_label, cfg.general.area_min)
    rasterized_data[defor_remove == 1] = 3

    target_test.GetRasterBand(1).WriteArray(rasterized_data)
    target_test = None
    
def generate_prev_map(cfg):
    
    base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]

    #previous_def_params = cfg['previous_def_params']

    f_yearly_def = Path(cfg.path.prodes.yearly_deforestation)
    v_yearly_def = ogr.Open(str(f_yearly_def))
    l_yearly_def = v_yearly_def.GetLayer()

    f_previous_def = Path(cfg.path.prodes.previous_deforestation)
    v_previous_def = ogr.Open(str(f_previous_def))
    l_previous_def = v_previous_def.GetLayer()

    base_data = gdal.Open(str(base_image), gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()


    delta_years = 10
    vals = np.linspace(0.1,1, delta_years)

    train_output = cfg.path.prev_map.train

    target_train = gdal.GetDriverByName('GTiff').Create(train_output, x_res, y_res, 1, gdal.GDT_Float32)
    target_train.SetGeoTransform(geo_transform)
    target_train.SetSpatialRef(crs)
    target_train.SetProjection(proj)

    train_year = cfg.general.year.train
    prev_train_year = train_year - 1

    years = np.linspace(prev_train_year-delta_years+1 ,prev_train_year, delta_years).astype(np.int32)


    gdal.RasterizeLayer(target_train, [1], l_previous_def, burn_values=[vals[0]])
    print('prev', vals[0])

    where = f'"year"<{years[0]}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[vals[0]])
    print(where, vals[0])

    for i, year in enumerate(years):
        v = vals[i]
        print(year, v)
        where = f'"year"={year}'
        l_yearly_def.SetAttributeFilter(where)
        gdal.RasterizeLayer(target_train, [1], l_yearly_def, burn_values=[v])

    target_train = None
        

    test_output = cfg.path.prev_map.test

    target_test = gdal.GetDriverByName('GTiff').Create(test_output, x_res, y_res, 1, gdal.GDT_Float32)
    target_test.SetGeoTransform(geo_transform)
    target_test.SetSpatialRef(crs)
    target_test.SetProjection(proj)

    test_year = cfg.general.year.test
    prev_test_year = test_year - 1

    years = np.linspace(prev_test_year-delta_years+1 ,prev_test_year, delta_years).astype(np.int32)


    gdal.RasterizeLayer(target_test, [1], l_previous_def, burn_values=[vals[0]])
    print('prev', vals[0])

    where = f'"year"<{years[0]}'
    l_yearly_def.SetAttributeFilter(where)
    gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[vals[0]])
    print(where, vals[0])

    for i, year in enumerate(years):
        v = vals[i]
        print(year, v)
        where = f'"year"={year}'
        l_yearly_def.SetAttributeFilter(where)
        gdal.RasterizeLayer(target_test, [1], l_yearly_def, burn_values=[v])

    target_test = None
    
def read_imgs(folder, imgs, read_fn, dtype, significance = 0, factor = 1.0, prefix_name = ''):
    pbar = tqdm(imgs, desc='Reading data')
    data = []
    for img_file in pbar:
        pbar.set_description(f'Reading {prefix_name}{img_file}')
        img_path = Path(folder) / f'{prefix_name}{img_file}'
        img = read_fn(img_path).astype(dtype)
        img = remove_outliers(img, significance)
        img = factor * img
        if len(img.shape) == 3:
            data.append(rearrange(img, 'h w c -> (h w) c'))
        elif len(img.shape) == 2:
            data.append(rearrange(img, 'h w -> (h w)'))
        
    return data
    
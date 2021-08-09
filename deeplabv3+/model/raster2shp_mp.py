from functools import partial
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import rasterio as rio
from rasterio import features
import shapely
from shapely import geometry, ops
import geopandas as gpd
# import time

from skimage import morphology
from skimage.morphology import remove_small_holes, remove_small_objects, disk

from regulization import coarse_adjustment, buffer

class_map = {1: 'cloud', 2: 'shadow', 3: 'road', 4: 'railway'}


def preprocess(img, min_spot=300):
    dtype = img.dtype
    assert dtype == np.uint8
    class_ids = np.unique(img)[1:]  # exclude zero value

    img_copy = img.copy()
    for class_id in class_ids:
        tmp_img = img == class_id
        # morphology.binary_opening(tmp_img, selem=disk(7), out=tmp_img)
        tmp_img = remove_small_objects(tmp_img, min_spot)
        tmp_img = remove_small_holes(tmp_img, min_spot)
        img_copy[tmp_img] = class_id

        # tmp_img = remove_small_objects(tmp_img, min_spot)
        # tmp_img = remove_small_holes(tmp_img, min_spot)


    return img_copy.astype(dtype)


def find_contours(image):
    assert image.dtype == np.uint8
    # Tuple(GeoJSON, value)
    shapes = features.shapes(image, mask=image != 0)
    results = [(int(v), geometry.shape(s)) for s, v in shapes]

    return results


def simplify(shapes):
    """Using DP algorithm to simplify polygon.
    """
    ret_shapes = []
    for class_id, geom in shapes:
        tol = 0.5 if geom.area < 10 else 2.0
        geom = geom.simplify(tol, preserve_topology=True)

        ret_shapes.append((class_id, geom))

    return ret_shapes


def polygonize(image, min_hole, min_object):
    """

    Return:
        List[tuple(dict, int)]: {GeoJSON format shapes: raster value}
    """
    # image = preprocess(image, min_hole, min_object)
    shapes = find_contours(image)
    shapes = simplify(shapes)

    return shapes


def tsf_coords(coords, tsf):
    ret = []
    for poly_coords in coords:
        cur = []
        for coord in poly_coords:
            cur.append(tsf * coord)
        ret.append(np.array(cur))

    return ret


def get_window_info(rfile, win_size=1000):
    window_list = []
    with rio.open(rfile) as src:
        height = src.height
        width = src.width
        for i in range(0, height, win_size - 3):
            if i + win_size < height:
                ysize = win_size
            else:
                ysize = height - i

            for j in range(0, width, win_size - 3):
                if j + win_size < width:
                    xsize = win_size
                else:
                    xsize = width - j

                window = rio.windows.Window(j, i, xsize, ysize)
                window_list.append(window)

    return window_list, src.crs, src.transform


def to_shapely(affine):
    """Return an affine transformation matrix compatible with shapely.

    (a,b,d,e,xoff,yoff)
    """

    return (affine.a, affine.b, affine.d, affine.e, affine.xoff, affine.yoff)


def extract_shapes(raster_file, window):
    results = []
    with rio.open(raster_file) as src:
        tsf = src.transform
        tsf = to_shapely(tsf)

        img_patch = src.read(1, window=window)
        x, y = window.col_off, window.row_off
        offset = np.array([x, y])
        shapes = polygonize(img_patch, 300, 300)
        for i, (class_id, geom) in enumerate(shapes, start=1):
            ext_coords = np.array(geom.exterior.coords) + offset
            int_coords = []
            if geom.interiors:
                for c in geom.interiors:
                    int_coords.append(np.array(c) + offset)
            geom = geometry.Polygon(ext_coords, int_coords)
            # geom = shapely.affinity.affine_transform(geom, tsf)

            results.append((class_id, geom))

    results = merge(results)

    return results


def merge(results):
    d = defaultdict(list)
    for v, geom in results:
        d[v].append(geom)

    ret = []
    for k, geoms in d.items():
        ret.append((k, ops.cascaded_union(geoms)))

    return ret


def explode(results):
    gdf = gpd.GeoDataFrame([{
        'class_id': v,
        'geometry': geom
    } for v, geom in results])
    geoms = gdf.explode()

    results = [(class_id, geom) for i, (class_id, geom) in geoms.iterrows()]
    return results
    # return geoms


def to_file(results, filename, tsf, crs, driver='ESRI Shapefile'):
    df = gpd.GeoDataFrame([{
        'class_id':
        v,
        'class_name':
        class_map[v],
        'geometry':
        shapely.affinity.affine_transform(geom, tsf)
    } for v, geom in results])
    if crs:
        df.crs = crs.to_dict()
    df.to_file(filename, driver=driver)


def ras2shp(rfile, shpfile):
    window_list, crs, tsf = get_window_info(rfile, win_size=1000)
    pool = mp.Pool(processes=4)
    extract_fn = partial(extract_shapes, rfile)
    results = pool.map(extract_fn, window_list)
    pool.close()
    pool.join()

    ret = []
    for r in results:
        ret.extend(r)

    results = merge(ret)

    results = explode(results)

    ret = []
    for class_id, geom in results:
        if class_id == 10:
            geom = coarse_adjustment(geom, alpha=30)
        else:
            geom = buffer(geom, size=10)
        if geom is not None:
            ret.append((class_id, geom))
    to_file(ret, shpfile, to_shapely(tsf), crs)

    # print(t1 - start)
    # print(t2 - t1)
    # print(t3 - t2)
    # print(t4 - t3)
    # print(f"Total time = {t4-start}")

# if __name__ == "__main__":
    # rfile = r"D:\Data\成都样本数据\xk1_gray.tif"
    # shpfile = r"D:\Data\成都样本数据\xk1_gray.shp"
    # ras2shp(rfile, shpfile)
import geopandas as gpd
import numpy as np
from shapely import geometry

from util import calc_length, calc_angle


def remove_short_edge(vertices, thresh=None):
    # N x 2
    vertices = vertices.copy()
    # N = len(vertices)
    lengths = calc_length(vertices)
    perimeter = lengths.sum()
    if thresh is None:
        thresh = perimeter * 0.002
    inds = np.argwhere(lengths >= thresh).squeeze()

    return vertices[inds]


def remove_angle(vertices, alpha=30, beta=170):
    vertices = vertices.copy()
    # N = len(vertices)
    angles = calc_angle(vertices)
    keep = (angles >= alpha) & (angles <= beta)

    return vertices[keep]


def buffer(geom, size=5):
    dilated = geom.buffer(size)
    geom = dilated.buffer(-size)
    tol = 0.5 if geom.area < 10 else 2.0
    geom = geom.simplify(tol, preserve_topology=True)
    return geom


def coarse_adjustment(geom,
                      area_low_thresh=10,
                      area_high_thresh=200,
                      alpha=30,
                      beta=170):
    """Building Regulization.

    1. Remove polygons whose area is below a threshold S.
    2. Remove edges whose lengths are below a given side length Td that varies with the area of a building.
    3. Remove over-sharp angles with threshold α.
    4. Remove over-smooth angles with threshold β.
    """
    geom_area = geom.area
    if geom_area <= area_low_thresh:
        return None
    # extract shell and holes
    # 起始点为重复的，首尾点相同，去掉尾部相同点
    shell = np.array(geom.exterior.coords)[:-1]
    holes = []
    holes_area = []
    if geom.interiors:
        for c in geom.interiors:
            area = geometry.Polygon(c).area
            holes_area.append(area)
            if area > area_low_thresh:
                holes.append(np.array(c)[:-1])

    # ==============================================
    # edge filter
    shell = remove_short_edge(shell)
    if len(shell) < 3:
        return None
    holes1 = []
    if holes:
        for h in holes:
            hole = remove_short_edge(h)
            if len(hole) >= 3:
                holes1.append(hole)

    # angle filter
    shell = remove_angle(shell)
    if len(shell) < 3:
        return None
    holes2 = []
    if holes:
        for h in holes1:
            hole = remove_angle(h)
            if len(hole) >= 3:
                holes2.append(hole)

    return geometry.Polygon(shell, holes2)


def fine_adjustment(geom):
    pass


# shp_df = gpd.read_file('building_test/original.shp')
# building_df = shp_df[shp_df.class_id == 1].filter(
#     items=['class_id', 'geometry'])
# building_df.to_file('building_test/building_original.shp')
def reg_building(ori_shp, reg_shp):
    building_df = gpd.read_file(ori_shp)

    results = []
    for i, (class_id, geom) in building_df.iterrows():
        new_geom = buffer(geom, size=5)
        new_geom = coarse_adjustment(new_geom)
        if new_geom is not None:
            results.append((class_id, new_geom))

    df = gpd.GeoDataFrame([{
        'class_id': v,
        'geometry': geom
    } for v, geom in results])

    df.to_file(reg_shp)


# if __name__ == "__main__":
#     ori_shp = 'building_test/building_original.shp'
#     reg_shp = 'building_test/building_regu.shp'
#     reg_building(ori_shp, reg_shp)

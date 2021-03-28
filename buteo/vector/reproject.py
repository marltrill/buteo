import sys; sys.path.append('../../')
from typing import Union
import osgeo; from osgeo import ogr, osr, gdal
from buteo.vector.io import (
    vector_to_reference,
    vector_to_memory,
    vector_to_metadata,
    vector_to_disk
)
from buteo.gdal_utils import parse_projection, path_to_driver
from buteo.utils import remove_if_overwrite


def reproject_vector(
    vector: Union[str, ogr.DataSource],
    projection: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference],
    out_path: Union[str, None]=None,
    overwrite: bool=True,
) -> Union[str, ogr.DataSource]:
    """ Reprojects a vector given a target projection.

    Args:
        vector (path | vector): The vector to reproject.
        
        projection (str | int | vector | raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, or read from a 
        vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An in-memory vector. If an out_path is given, the output is a string containing
        the path to the newly created vecotr.
    """
    origin = vector_to_reference(vector)
    metadata = vector_to_metadata(origin)

    origin_projection = metadata["projection_osr"]
    target_projection = parse_projection(projection)

    if origin_projection.IsSame(target_projection):
        if out_path is None:
            return vector_to_memory(vector)
        
        return vector_to_disk(vector, out_path)

    remove_if_overwrite(out_path, overwrite)

    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        origin_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_trans = osr.CoordinateTransformation(origin_projection, target_projection)

    driver = None
    destination = None
    if out_path is not None:
        driver = ogr.GetDriverByName(path_to_driver(out_path))
        destination = driver.CreateDataSource(out_path)
    else:
        driver = ogr.GetDriverByName('Memory')
        destination = driver.CreateDataSource(metadata["name"])

    for layer_idx in range(len(metadata["layers"])):
        origin_layer = origin.GetLayerByIndex(layer_idx)
        origin_layer_defn = origin_layer.GetLayerDefn()

        layer_dict = metadata["layers"][layer_idx]
        layer_name = layer_dict["layer_name"]
        layer_geom_type = layer_dict["geom_type_ogr"]

        destination_layer = destination.CreateLayer(layer_name, target_projection, layer_geom_type)
        destination_layer_defn = destination_layer.GetLayerDefn()

        # Copy field definitions
        origin_layer_defn = origin_layer.GetLayerDefn()
        for i in range(0, origin_layer_defn.GetFieldCount()):
            field_defn = origin_layer_defn.GetFieldDefn(i)
            destination_layer.CreateField(field_defn)

        # Loop through the input features
        for _ in range(origin_layer.GetFeatureCount()):
            feature = origin_layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)

            new_feature = ogr.Feature(destination_layer_defn)
            new_feature.SetGeometry(geom)

            # Copy field values
            for i in range(0, destination_layer_defn.GetFieldCount()):
                new_feature.SetField(destination_layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))

            destination_layer.CreateFeature(new_feature)
            
        destination_layer.ResetReading()
        destination_layer = None

    if out_path is not None:
        destination = None
        return out_path
    else:
        return destination


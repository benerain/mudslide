# -*- coding: utf-8 -*-
"""
@Project : PIEDaShuJuYaoSu 
@File    : main.py
@Time    : 2022/12/9 09:41
@Author  : liyujia
@Desc    : 泥石流识别算法
"""
import os
from osgeo import gdal, ogr, osr
import numpy as np
import sys
import cv2
import argparse
import logging
import json
import shapely.wkt
import shapely.geometry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

gdal.UseExceptions()

def getImageBounds(tif_file):
    try:
        dataset = gdal.Open(tif_file)
    except Exception as e:
        logger.error(f"open file error: {e}")
        return

    if dataset is None:
        logger.error(f"Failed to use gdal tif band file: {tif_file}")
        return

    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize
    geoTransform = dataset.GetGeoTransform()
    xMin = geoTransform[0]
    yMax = geoTransform[3]
    xCell = geoTransform[1]
    yCell = geoTransform[5]
    xMax = xMin + xCell * nXSize
    yMin = yMax + yCell * nYSize

    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjectionRef())
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    coordTrans = osr.CoordinateTransformation(source, target)
    transXY = coordTrans.TransformPoint(xMin, yMin)
    xMin = transXY[1]
    yMin = transXY[0]
    transXY = coordTrans.TransformPoint(xMax, yMax)
    xMax = transXY[1]
    yMax = transXY[0]
    bounds = f"POLYGON (({xMin} {yMin},{xMax} {yMin},{xMax} {yMax},{xMin} {yMax},{xMin} {yMin}))"
    return bounds

def getIntersectFile(region1, region2, dest_file):
    g1 = shapely.wkt.loads(region1)
    g2 = shapely.wkt.loads(region2)
    if not g1.intersects(g2):
        logger.error("two region not intersect")
        return
    result = g1.intersection(g2)
    roi = shapely.geometry.mapping(result)
    content = {
        "type": "FeatureCollection",
        "name": "bounds",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": [
            { "type": "Feature", "properties": { "ID":0 }, "geometry": roi }
        ]
    }
    if os.path.exists(dest_file):
        os.remove(dest_file)
    dest_path = os.path.dirname(dest_file)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(dest_file, "w+") as f:
        f.write(json.dumps(content))

def clipTifByShp(src_file, dest_file, shp):
    """
    裁剪单个文件
    :param src_file:
    :param dest_file:
    :param shp:
    :return:
    """
    if not os.path.exists(src_file):
        return
    if os.path.exists(dest_file):
        os.remove(dest_file)
    dest_path = os.path.dirname(dest_file)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    cmd = "gdalwarp -co COMPRESS=LZW -co BIGTIFF=YES {inputTif} {outputTif} -cutline {shapeFile} -crop_to_cutline".format(
        inputTif=src_file,
        outputTif=dest_file,
        shapeFile=shp
    )
    logger.info(f"clip raster tif command is : {cmd}")
    os.system(cmd)

def getRasterBandInfo(rasterFileName, bandNumber=1):
    """
    获得影像的信息
    :param rasterFileName:
    :param bandNumber:
    :return:
    """
    try:
        raster = gdal.Open(rasterFileName)
        band = raster.GetRasterBand(bandNumber)
        geotransform = raster.GetGeoTransform()
        noDataValue = band.GetNoDataValue()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        rtnX = geotransform[2]
        rtnY = geotransform[4]
        cols = raster.RasterXSize
        rows = raster.RasterYSize
        logger.info("=============rasterFileName = {0} ================".format(rasterFileName))
        logger.info("noDataValue = {}".format(noDataValue))
        logger.info("rtnX = {0}, rtnY = {1}".format(rtnX, rtnY))
        logger.info("originX = {0}, originY = {1}".format(originX, originY))
        logger.info("pixelWidth = {0}, pixelHeight = {1}".format(pixelWidth, pixelHeight))
        logger.info("cols = {0}, rows = {1}".format(cols, rows))
        bandArray = band.ReadAsArray()
        logger.info("=======================end==========================")
        return {
            "array": bandArray,
            "geotransform": geotransform,
            "cols": cols,
            "rows": rows,
            "pixelWidth": pixelWidth,
            "pixelHeight": pixelHeight,
            "nodata": noDataValue
        }
    except RuntimeError as e:
        logger.error("can not open rasterFile : {0}, error is {1}".format(rasterFileName, e))
        return None

def generateSingleTifByArray(in_tif, in_array, out_tif, invalid_value=None, dtype=gdal.GDT_UInt16):
    """
    通过数组生成tif
    :param in_tif:
    :param in_array:
    :param out_tif:
    :param invalid_value:
    :param dtype:
    :return:
    """
    if os.path.exists(out_tif):
        os.remove(out_tif)

    out_tif_path = os.path.dirname(out_tif)
    if not os.path.exists(out_tif_path):
        os.makedirs(out_tif_path)
    try:
        raster = gdal.Open(in_tif)
    except RuntimeError as e:
        logger.info("can not open rasterFile : {0}, error is {1}".format(in_tif, e))
        return False
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_tif, cols, rows, 1, dtype, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    if invalid_value is not None:
        outband.Fill(invalid_value)
        outband.SetNoDataValue(invalid_value)

    xBlockSize = 256
    yBlockSize = 256
    for i in range(0, rows, yBlockSize):
        if i + yBlockSize < rows:
            numRows = yBlockSize
        else:
            numRows = rows - i
        for j in range(0, cols, xBlockSize):
            if j + xBlockSize < cols:
                numCols = xBlockSize
            else:
                numCols = cols - j
            resultArray = in_array[i:i + numRows, j:j + numCols]
            outband.WriteArray(resultArray, j, i)
    outband.FlushCache()

def preproImage(pre_file, new_file):
    pre_bounds = getImageBounds(pre_file)
    new_bounds = getImageBounds(new_file)
    if pre_bounds == new_bounds:
        logger.info("two image bounds is same")
        return True
    _pre_path = os.path.dirname(pre_file)
    _pre_name = os.path.splitext(os.path.basename(pre_file))[0]
    _new_path = os.path.dirname(new_file)
    _new_name = os.path.splitext(os.path.basename(new_file))[0]
    _dest_geojson = os.path.join(_pre_path, f"{_pre_name}_{_new_name}.geojson")
    if os.path.exists(_dest_geojson):
        os.remove(_dest_geojson)
    getIntersectFile(pre_bounds, new_bounds, _dest_geojson)
    if not os.path.exists(_dest_geojson):
        logger.error("not find intersection")
        return False
    _pre_temp_file = os.path.join(_pre_path, f"temp-{_pre_name}.tif")
    clipTifByShp(pre_file, _pre_temp_file, _dest_geojson)
    if os.path.exists(_pre_temp_file):
        os.remove(pre_file)
        os.system(f"mv {_pre_temp_file} {pre_file}")
    _new_temp_file = os.path.join(_new_path, f"temp-{_new_name}.tif")
    clipTifByShp(new_file, _new_temp_file, _dest_geojson)
    if os.path.exists(_new_temp_file):
        os.remove(new_file)
        os.system(f"mv {_new_temp_file} {new_file}")
    if os.path.exists(_dest_geojson):
        os.remove(_dest_geojson)
    return True

def polygonizeRaster(tif_file, shp_file):
    try:
        raster = gdal.Open(tif_file)
        srcband = raster.GetRasterBand(1)
        maskband = srcband.GetMaskBand()
    except RuntimeError as e:
        logger.error("can not open rasterFile : {0}, error is {1}".format(tif_file, e))
        return

    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(shp_file)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjectionRef())
    dst_layername = 'mudrock'
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)

    dst_fieldname = 'DN'
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0

    options = []
    # 参数  输入栅格图像波段\掩码图像波段、矢量化后的矢量图层、需要将DN值写入矢量字段的索引、算法选项、进度条回调函数、进度条参数
    gdal.Polygonize(srcband, maskband, dst_layer, dst_field, options)

def generateVector(outFileshpPath, outFiletifPath):
    _dest_path = os.path.dirname(outFileshpPath)
    _dest_name = os.path.splitext(os.path.basename(outFileshpPath))[0]
    os.system(f"rm -rf {_dest_path}/temp-{_dest_name}*")
    temp_shp1 = os.path.join(_dest_path, f"temp-{_dest_name}1.shp")
    logger.info(f"generate polygon")
    if os.path.exists(outFiletifPath):
        polygonizeRaster(outFiletifPath, temp_shp1)

    temp_shp2 = os.path.join(_dest_path, f"temp-{_dest_name}2.shp")
    if os.path.exists(temp_shp1):
        os.system(f"ogr2ogr -t_srs EPSG:3857 {temp_shp2} {temp_shp1}")

    logger.info(f"filter polygon")
    temp_shp3 = os.path.join(_dest_path, f"temp-{_dest_name}3.shp")
    if os.path.exists(temp_shp2):
        try:
            dataSource = ogr.Open(temp_shp2)
        except Exception as e:
            logger.info("open shape file error, error is: {0}".format(e))
            return None
        layer = dataSource.GetLayer(0)

        driver = ogr.GetDriverByName("ESRI Shapefile")
        newds = driver.CreateDataSource(temp_shp3)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        newLayer = newds.CreateLayer("mudRock", srs=srs, geom_type=ogr.wkbMultiPolygon)
        lyr_def = layer.GetLayerDefn()
        for i in range(lyr_def.GetFieldCount()):
            newLayer.CreateField(lyr_def.GetFieldDefn(i))

        # 创建新的字段 面积，用于去除过小的斑块
        new_field = ogr.FieldDefn("Area", ogr.OFTReal)
        new_field.SetWidth(32)
        new_field.SetPrecision(16)
        newLayer.CreateField(new_field)

        feature = layer.GetNextFeature()
        while feature:
            geom = feature.geometry()
            keys = feature.keys()
            area_in_sq_m = geom.GetArea()
            if area_in_sq_m <= 100:
                feature = layer.GetNextFeature()
                continue

            out_feat = ogr.Feature(newLayer.GetLayerDefn())
            out_feat.SetGeometry(geom)

            out_feat.SetField("Area", area_in_sq_m)
            for key in keys:
                out_feat.SetField(key, feature.GetField(key))
            newLayer.CreateFeature(out_feat)
            newLayer.SyncToDisk()
            feature = layer.GetNextFeature()
        dataSource.Destroy()

    logger.info(f"generate result polygon")
    if os.path.exists(temp_shp3):
        os.system(f"ogr2ogr -t_srs EPSG:4326 {outFileshpPath} {temp_shp3}")
    os.system(f"rm -rf {_dest_path}/temp-{_dest_name}*")

class MudRockFlowTool():
    """
    A : a class to detect landslide
    """
    def _elegentNpDivide(self, up, down):
        return np.divide(up, down, out=np.zeros(up.shape), where=down != 0)

    def _calNdvi(self, ds, n, r):
        b_n = ds.GetRasterBand(n).ReadAsArray().astype(np.float64)
        b_r = ds.GetRasterBand(r).ReadAsArray().astype(np.float64)
        up = b_n - b_r
        down = b_n + b_r
        return self._elegentNpDivide(up, down)

    def redChange(self, dataset_hist, dataset_new, r):
        # r: int 代表红色波段在第几个波段, 从1开始计数
        ds_hist = dataset_hist.GetRasterBand(r).ReadAsArray().astype(np.float64)
        ds_new = dataset_new.GetRasterBand(r).ReadAsArray().astype(np.float64)

        change = ds_new - ds_hist
        redchange = self._elegentNpDivide(change, ds_hist)

        redchangemask = np.where((redchange > 0.4) & (redchange < 2), 1, 0)

        return redchangemask  # np.array

    def _calNdwi(self, ds, g, n):
        b_g = ds.GetRasterBand(g).ReadAsArray().astype(np.float64)
        b_n = ds.GetRasterBand(n).ReadAsArray().astype(np.float64)

        up = b_g - b_n
        down = b_g + b_n

        return self._elegentNpDivide(up, down)

    def ndwiMask(self, dataset_hist, dataset_new, g, n):
        # g: int 代表绿色波段在第几个波段, 从1开始计数
        # n: int 代表近红外波段在第几个波段, 从1开始计数
        ndwi_hist = self._calNdwi(dataset_hist, g, n)
        ndwi_new = self._calNdwi(dataset_new, g, n)

        ndwiMask = np.where((ndwi_hist < 0.5) | (ndwi_new < 0.5), 1, 0)

        return ndwiMask  # np.array

    def ndviLoss(self, dataset_hist, dataset_new, n, r):
        # n: int 代表近红外波段在第几个波段, 从1开始计数
        # r: int 代表红色波段在第几个波段 从1开始计数
        ndwi_hist = self._calNdvi(dataset_hist, n, r)
        ndwi_new = self._calNdvi(dataset_new, n, r)
        ndviChangeMask = np.where((ndwi_hist > ndwi_new), 1, 0)
        return ndviChangeMask  # np.array

    def window_sum(self,array,size):
        table = np.cumsum(np.cumsum(array, axis=0), axis=1)
        win_sum = np.empty(tuple(np.subtract(array.shape, size - 1)))
        win_sum[0, 0] = table[size - 1, size - 1]
        win_sum[0, 1:] = table[size - 1, size:] - table[size - 1, :-size]
        win_sum[1:, 0] = table[size:, size - 1] - table[:-size, size - 1]
        win_sum[1:, 1:] = (table[size:, size:] + table[:-size, :-size] -
                           table[size:, :-size] - table[:-size, size:])
        return win_sum

    def texture (self,a ,win = 3):
        # win: int 必须是奇数
        win = 3 if win < 3 else win

        a_pad = np.pad(a, int((win - 1) / 2), 'constant', constant_values=0)
        win_a = self.window_sum(a_pad, win)
        win_a2 = self.window_sum(a_pad * a_pad, win)
        return (win_a2 - win_a * win_a / win / win) / win / win

    def textureChange(self, dataset_hist, dataset_new, i, win):
        # i: int 代表使用第几个波段进行灰度图像的纹理检测, 从1开始计数
        # win: int 代表kernel的边长，必须是正奇数
        ds_hist = dataset_hist.GetRasterBand(i).ReadAsArray().astype(np.float64)
        ds_new = dataset_new.GetRasterBand(i).ReadAsArray().astype(np.float64)

        texture_hist = self.texture(ds_hist,win)
        texture_new = self.texture(ds_new,win)

        texture_change = texture_new - texture_hist
        texture_mask = np.where((texture_change < - 10), 1, 0)

        return texture_mask

    def calcLandslide(self, dataset_hist_Path, dataset_new_Path, outFiletifPath,outFileshpPath):
        logger.info("start process")
        # 打开输入数据集
        dataset_hist = gdal.Open(dataset_hist_Path, gdal.GA_ReadOnly)
        dataset_new = gdal.Open(dataset_new_Path, gdal.GA_ReadOnly)

        # 验证数据是否正确打开
        if dataset_hist is None or dataset_new is None:
            logger.info("输入数据无法打开")
            sys.exit(-1)

        logger.info("find mud-rock flow")
        # 1、检测红移
        redchange = self.redChange(dataset_hist, dataset_new, 1)
        # 2、去除水体
        watermask = self.ndwiMask(dataset_hist, dataset_new, 2, 4)
        # 3、检测植被的减少
        ndviloss =self.ndviLoss(dataset_hist, dataset_new, 4, 1)
        # 4、检测纹理变化
        texturechange = self.textureChange(dataset_hist, dataset_new, 4, 7)
        detection = redchange + watermask + ndviloss + texturechange
        mask = np.where((detection == 4), 1, 0)
        # 形态学处理
        kernel1 = np.ones((5, 5), np.uint8)
        mask_ero = cv2.erode(mask.astype('uint8'), kernel1, iterations=1)
        kernel2 = np.ones((10, 10), np.uint8)
        mask_exp = cv2.dilate(mask_ero.astype('uint8'), kernel2, iterations=5)

        # 输出图像
        logger.info("generate tif file")
        generateSingleTifByArray(in_tif=dataset_hist_Path,
                                 in_array=mask_exp,
                                 out_tif=outFiletifPath,
                                 invalid_value=0,
                                 dtype=gdal.GDT_Byte)
        
        logger.info("generate polygon")
        # 输出矢量
        generateVector(outFileshpPath, outFiletifPath)

        logger.info('process finish')

def main(pre_image, new_image, dest_shp):
    if not os.path.exists(pre_image):
        logger.error(f"{pre_image} is not exist!")
        return
    if not os.path.exists(new_image):
        logger.error(f"{new_image} is not exist!")
        return
    _dest_path = os.path.dirname(dest_shp)
    if not os.path.exists(_dest_path):
        os.makedirs(_dest_path)
    _dest_name = os.path.splitext(os.path.basename(dest_shp))[0]
    os.system(f"rm -rf {_dest_path}/{_dest_name}*")

    # 预处理影像
    flag = preproImage(pre_image, new_image)
    if not flag:
        logger.error(f"preprocess image fail!")
        return

    tool = MudRockFlowTool()
    # 进行滑坡计算
    dest_tif = os.path.join(_dest_path, f"{_dest_name}.tif")
    if os.path.exists(dest_tif):
        os.remove(dest_tif)
    tool.calcLandslide(pre_image, new_image, dest_tif, dest_shp)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='参数')
    parser.add_argument("-pre","--pre_image" ,action="store",
                        dest="pre_image",
                        default=None,
                        help="灾害前影像文件")
    parser.add_argument("-new", "--new_image", action="store",
                        dest="new_image",
                        default=None,
                        help="灾害后影像文件")
    parser.add_argument("-o", "--output", action="store",
                        dest="output", default=None,
                        help="输出的矢量文件")
    args = parser.parse_args()

    _pre = args.pre_image
    _new = args.new_image
    _output = args.output

    if not _pre:
        raise Exception("灾害前原始文件不能为空")
    if not _new:
        raise Exception("灾害后原始文件不能为空")
    if not _output:
        raise Exception("输出文件不能为空")
    main(_pre, _new, _output)
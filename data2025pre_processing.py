#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
from contextlib import ExitStack

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge as raster_merge
from shapely.geometry import shape

BASE_DIR = Path(r"D:\GisHomwork")
DEFAULT_INPUTS = [
    BASE_DIR / "RawData" / "Mangrove_Extraction_2025-0000000000-0000000000 (1).tif",
    BASE_DIR / "RawData" / "Mangrove_Extraction_2025-0000000000-0000032768.tif",
]
DEFAULT_OUTPUT = BASE_DIR / "RawData" / "Mangrove_Extraction_2025_vector.shp"
DEFAULT_TARGET_VALUE = 1
DEFAULT_YEAR = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 GEE 导出的 2025 红树林分类栅格先镶嵌，再提取值 1 并转换为矢量面。"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="输入栅格路径列表，必须是 .tif 文件，不要使用 .ovr 文件。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出矢量路径，建议为 .shp 文件。",
    )
    parser.add_argument(
        "--target-value",
        type=int,
        default=DEFAULT_TARGET_VALUE,
        help="需要提取为红树林面的像元值，默认 1。",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="输出属性字段 Year 的值，默认 2025。",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="是否将所有相邻/相交红树林面融合为一个或多个大面。",
    )
    return parser.parse_args()


def validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    if input_path.suffix.lower() != ".tif":
        raise ValueError(
            f"输入文件必须是 .tif 栅格，当前为: {input_path.name}。"
            "`.ovr` 是金字塔概览文件，不能作为主栅格转换。"
        )


def validate_inputs(input_paths: list[Path]) -> None:
    if not input_paths:
        raise ValueError("至少需要提供一个 .tif 输入文件。")
    for input_path in input_paths:
        validate_input(input_path)


def remove_existing_shapefile(output_path: Path) -> None:
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        sidecar = output_path.with_suffix(ext)
        if sidecar.exists():
            sidecar.unlink()


def mosaic_rasters(input_paths: list[Path]):
    validate_inputs(input_paths)

    print("准备镶嵌以下栅格：")
    for input_path in input_paths:
        print(f"  - {input_path}")

    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(input_path)) for input_path in input_paths]

        reference_crs = datasets[0].crs
        reference_count = datasets[0].count
        nodata = datasets[0].nodata

        for dataset in datasets[1:]:
            if dataset.crs != reference_crs:
                raise ValueError("输入栅格坐标系不一致，无法直接镶嵌。")
            if dataset.count != reference_count:
                raise ValueError("输入栅格波段数不一致，无法直接镶嵌。")

        mosaic_array, mosaic_transform = raster_merge(datasets, method="first")
        band1 = mosaic_array[0]

    print(f"镶嵌完成，输出尺寸: {band1.shape[1]} x {band1.shape[0]}")
    print(f"坐标系: {reference_crs}")
    print(f"nodata: {nodata}")
    return band1, mosaic_transform, reference_crs, nodata


def polygonize_mangrove(
    input_paths: list[Path],
    output_path: Path,
    target_value: int,
    year: int,
    dissolve: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    band1, transform, crs, nodata = mosaic_rasters(input_paths)

    if nodata is None:
        valid_mask = np.ones(band1.shape, dtype=bool)
    else:
        valid_mask = band1 != nodata

    target_mask = (band1 == target_value) & valid_mask
    target_pixels = int(target_mask.sum())
    print(f"目标像元值 {target_value} 的像元数: {target_pixels}")

    if target_pixels == 0:
        raise ValueError(
            f"栅格中没有值为 {target_value} 的像元，无法生成红树林矢量面。"
        )

    geometry_records = []
    for geom, value in shapes(band1, mask=target_mask, transform=transform):
        if int(value) != target_value:
            continue
        geometry_records.append(
            {
                "geometry": shape(geom),
                "gridcode": int(value),
                "PXLVAL": 1,
                "Year": year,
            }
        )

    if not geometry_records:
        raise ValueError("未提取到任何有效面。")

    gdf = gpd.GeoDataFrame(geometry_records, crs=crs)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    print(f"初始矢量面数量: {len(gdf)}")

    if dissolve:
        print("执行融合 dissolve...")
        merged = gdf.dissolve(by="PXLVAL", as_index=False)
        merged["Year"] = year
        merged["gridcode"] = target_value
        gdf = merged[["gridcode", "PXLVAL", "Year", "geometry"]].copy()
        print(f"融合后面数量: {len(gdf)}")

    remove_existing_shapefile(output_path)
    gdf.to_file(output_path, encoding="utf-8")
    print(f"输出完成: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    polygonize_mangrove(
        input_paths=args.inputs,
        output_path=args.output,
        target_value=args.target_value,
        year=args.year,
        dissolve=args.dissolve,
    )


if __name__ == "__main__":
    main()

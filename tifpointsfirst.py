#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import ExitStack
from pathlib import Path
import random

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import bounds as window_bounds
from shapely import make_valid
from shapely.geometry import Point

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(r"D:\GisHomwork\RawData")
FISHNET_PATH = BASE_DIR / "fishnet_top62" / "fishnet_top62.shp"
OUTPUT_DIR = BASE_DIR / "output_samples"
TARGET_CRS = "EPSG:4326"
TARGET_FIDS = [48, 49]
SAMPLES_PER_GRID = 1000
TARGET_VALUE = 1
TARGET_YEAR = 2025

YEAR_SAMPLE_PATHS = {
    2010: OUTPUT_DIR / "samples_2010.shp",
    2015: OUTPUT_DIR / "samples_2015.shp",
    2020: OUTPUT_DIR / "samples_2020.shp",
    2025: OUTPUT_DIR / "samples_2025.shp",
}

MERGED_OUTPUT_PATH = OUTPUT_DIR / "all_samples_2010_2025.shp"


def ensure_path_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")


def clear_shapefile(path: Path) -> None:
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        sidecar = path.with_suffix(ext)
        if sidecar.exists():
            sidecar.unlink()


def save_gdf(gdf: gpd.GeoDataFrame, path: Path) -> None:
    clear_shapefile(path)
    gdf.to_file(path, index=False)


def clean_polygon_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return gdf

    invalid_mask = ~gdf.geometry.is_valid
    if int(invalid_mask.sum()) > 0:
        try:
            gdf["geometry"] = gdf.geometry.apply(make_valid)
        except Exception:
            gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return gdf

    still_invalid_mask = ~gdf.geometry.is_valid
    if int(still_invalid_mask.sum()) > 0:
        gdf = gdf[~still_invalid_mask].copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def discover_raster_paths() -> list[Path]:
    raster_paths = sorted(BASE_DIR.glob("Mangrove_Extraction_2025-*.tif"))
    if not raster_paths:
        raise FileNotFoundError("未找到 2025 分类 tif 文件。")

    print("检测到以下 2025 栅格分片：")
    for raster_path in raster_paths:
        print(f"  - {raster_path}")

    return raster_paths


def select_target_grids() -> gpd.GeoDataFrame:
    print("步骤1：提取目标格网（FID=48,49）")
    fishnet = gpd.read_file(FISHNET_PATH)

    if fishnet.crs is None:
        raise ValueError("fishnet_top62.shp 缺少坐标系定义，无法继续处理。")

    fid_field = None
    for field in ["FID", "fid", "OBJECTID", "OID", "Id"]:
        if field in fishnet.columns:
            fid_field = field
            break

    if fid_field:
        target_grids = fishnet[fishnet[fid_field].astype(int).isin(TARGET_FIDS)].copy()
        target_grids["src_fid"] = target_grids[fid_field].astype(int)
    else:
        if len(fishnet) <= max(TARGET_FIDS):
            raise IndexError("fishnet 要素数量不足，无法按行索引提取 48 和 49。")
        target_grids = fishnet.iloc[TARGET_FIDS].copy()
        target_grids["src_fid"] = TARGET_FIDS

    target_grids = target_grids.sort_values("src_fid").copy()
    if len(target_grids) != 2 or set(target_grids["src_fid"].tolist()) != set(TARGET_FIDS):
        raise ValueError("目标格网提取失败，未能准确获取 FID=48 和 FID=49 两个格网。")

    if "grid_id" in target_grids.columns:
        target_grids["src_gid"] = target_grids["grid_id"]
    else:
        target_grids["src_gid"] = None

    target_grids["grid_id"] = target_grids["src_fid"].astype(int)
    target_grids = clean_polygon_gdf(target_grids)
    target_grids = target_grids.to_crs(TARGET_CRS)
    target_grids = target_grids[["grid_id", "src_fid", "src_gid", "geometry"]].copy()

    print("目标格网映射关系：")
    for _, row in target_grids.iterrows():
        print(
            f"  FID={row['src_fid']}, 原始grid_id={row['src_gid']}, 输出grid_id={row['grid_id']}"
        )

    return target_grids


def bbox_intersects(bounds1, bounds2) -> bool:
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2
    return not (maxx1 <= minx2 or maxx2 <= minx1 or maxy1 <= miny2 or maxy2 <= miny1)


def collect_candidate_windows(dataset, polygon_geom, target_value: int) -> tuple[list[dict], int]:
    polygon_bounds = polygon_geom.bounds
    nodata = dataset.nodata
    candidate_windows = []
    total_pixels = 0

    for _, window in dataset.block_windows(1):
        current_bounds = window_bounds(window, dataset.transform)
        if not bbox_intersects(current_bounds, polygon_bounds):
            continue

        band = dataset.read(1, window=window)
        transform = dataset.window_transform(window)
        inside_mask = geometry_mask(
            [polygon_geom],
            out_shape=band.shape,
            transform=transform,
            invert=True,
        )

        valid_mask = inside_mask & (band == target_value)
        if nodata is not None:
            valid_mask &= band != nodata

        count = int(valid_mask.sum())
        if count == 0:
            continue

        candidate_windows.append(
            {
                "window": window,
                "count": count,
            }
        )
        total_pixels += count

    return candidate_windows, total_pixels


def build_points_for_grid(datasets, grid_geometry, grid_id: int, n_points: int) -> list[Point]:
    raster_crs = datasets[0].crs
    grid_geom_raster_crs = gpd.GeoSeries([grid_geometry], crs=TARGET_CRS).to_crs(raster_crs).iloc[0]

    weighted_windows = []
    total_pixels = 0
    for dataset_index, dataset in enumerate(datasets):
        current_windows, pixel_count = collect_candidate_windows(dataset, grid_geom_raster_crs, TARGET_VALUE)
        for window_info in current_windows:
            weighted_windows.append(
                {
                    "dataset_index": dataset_index,
                    "window": window_info["window"],
                    "count": window_info["count"],
                }
            )
        total_pixels += pixel_count

    if total_pixels == 0:
        raise ValueError(f"2025 年 grid_id={grid_id} 范围内没有值为 1 的像元，无法生成样本。")

    print(f"  grid_id={grid_id} 可采样像元数：{total_pixels}")
    print(f"  grid_id={grid_id} 候选窗口数：{len(weighted_windows)}")

    probabilities = np.array([info["count"] for info in weighted_windows], dtype=float)
    probabilities = probabilities / probabilities.sum()
    chosen_window_indices = np.random.choice(len(weighted_windows), size=n_points, p=probabilities)

    window_cache = {}
    points = []

    for chosen_index in chosen_window_indices:
        window_info = weighted_windows[int(chosen_index)]
        dataset = datasets[window_info["dataset_index"]]
        window = window_info["window"]
        cache_key = (
            int(window_info["dataset_index"]),
            int(window.col_off),
            int(window.row_off),
            int(window.width),
            int(window.height),
        )

        if cache_key not in window_cache:
            band = dataset.read(1, window=window)
            transform = dataset.window_transform(window)
            inside_mask = geometry_mask(
                [grid_geom_raster_crs],
                out_shape=band.shape,
                transform=transform,
                invert=True,
            )
            valid_mask = inside_mask & (band == TARGET_VALUE)
            if dataset.nodata is not None:
                valid_mask &= band != dataset.nodata

            rows, cols = np.where(valid_mask)
            if len(rows) == 0:
                continue
            window_cache[cache_key] = (rows, cols, transform)

        rows, cols, transform = window_cache[cache_key]
        selected_pixel_index = random.randrange(len(rows))
        row = int(rows[selected_pixel_index])
        col = int(cols[selected_pixel_index])

        center_x, center_y = rasterio.transform.xy(transform, row, col, offset="center")
        point = Point(center_x, center_y)
        points.append(point)

    if len(points) != n_points:
        raise RuntimeError(
            f"grid_id={grid_id} 样本点生成失败：目标 {n_points} 个，实际生成 {len(points)} 个。"
        )

    points_gdf = gpd.GeoDataFrame(geometry=points, crs=raster_crs)
    points_gdf = points_gdf.to_crs(TARGET_CRS)
    return points_gdf.geometry.tolist()


def validate_samples(samples: gpd.GeoDataFrame, year: int) -> None:
    required_fields = {"grid_id", "Year", "PXLVAL", "geometry"}
    if not required_fields.issubset(set(samples.columns)):
        raise ValueError(f"{year} 年样本缺少必要字段：{required_fields - set(samples.columns)}")

    counts = samples["grid_id"].value_counts().to_dict()
    for grid_id in TARGET_FIDS:
        if counts.get(grid_id, 0) != SAMPLES_PER_GRID:
            raise ValueError(
                f"{year} 年 grid_id={grid_id} 样本数错误：应为 {SAMPLES_PER_GRID}，实际为 {counts.get(grid_id, 0)}"
            )

    if not (samples["Year"] == year).all():
        raise ValueError(f"{year} 年样本 Year 字段存在异常值。")

    if not (samples["PXLVAL"] == 1).all():
        raise ValueError(f"{year} 年样本 PXLVAL 字段存在非 1 的值。")

    if len(samples) != SAMPLES_PER_GRID * len(TARGET_FIDS):
        raise ValueError(
            f"{year} 年总样本数错误：应为 {SAMPLES_PER_GRID * len(TARGET_FIDS)}，实际为 {len(samples)}"
        )


def generate_2025_samples(target_grids: gpd.GeoDataFrame, raster_paths: list[Path]) -> gpd.GeoDataFrame:
    print("步骤2：从 2025 分类 tif 中直接采样")

    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(raster_path)) for raster_path in raster_paths]

        if datasets[0].crs is None:
            raise ValueError("2025 tif 缺少坐标系定义。")

        base_crs = datasets[0].crs
        for dataset in datasets[1:]:
            if dataset.crs != base_crs:
                raise ValueError("输入 tif 坐标系不一致，无法联合采样。")

        year_samples = []
        for _, grid_row in target_grids.iterrows():
            grid_id = int(grid_row["grid_id"])
            points = build_points_for_grid(datasets, grid_row.geometry, grid_id, SAMPLES_PER_GRID)
            sample_gdf = gpd.GeoDataFrame(
                {
                    "grid_id": [grid_id] * SAMPLES_PER_GRID,
                    "Year": [TARGET_YEAR] * SAMPLES_PER_GRID,
                    "PXLVAL": [1] * SAMPLES_PER_GRID,
                },
                geometry=points,
                crs=TARGET_CRS,
            )
            year_samples.append(sample_gdf)
            print(f"  grid_id={grid_id} 已生成 {len(sample_gdf)} 个点")

    samples_2025 = gpd.GeoDataFrame(pd.concat(year_samples, ignore_index=True), crs=TARGET_CRS)
    validate_samples(samples_2025, TARGET_YEAR)
    return samples_2025


def merge_all_samples() -> gpd.GeoDataFrame:
    print("步骤3：合并 2010/2015/2020/2025 四年样本")

    merge_order = [2010, 2015, 2020, 2025]
    merged_parts = []

    for year in merge_order:
        sample_path = YEAR_SAMPLE_PATHS[year]
        ensure_path_exists(sample_path)
        gdf = gpd.read_file(sample_path)

        if gdf.crs is None:
            raise ValueError(f"{sample_path.name} 缺少坐标系定义。")

        gdf = gdf.to_crs(TARGET_CRS)
        required_columns = ["grid_id", "Year", "PXLVAL", "geometry"]
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            raise ValueError(f"{sample_path.name} 缺少字段：{missing_columns}")

        expected_count = SAMPLES_PER_GRID * len(TARGET_FIDS)
        if len(gdf) != expected_count:
            raise ValueError(
                f"{sample_path.name} 样本数错误：应为 {expected_count}，实际为 {len(gdf)}"
            )

        merged_parts.append(gdf[required_columns].copy())
        print(f"  已加入：{sample_path}")

    merged_gdf = gpd.GeoDataFrame(
        pd.concat(merged_parts, ignore_index=True),
        crs=TARGET_CRS,
    )

    expected_total = SAMPLES_PER_GRID * len(TARGET_FIDS) * len(merge_order)
    if len(merged_gdf) != expected_total:
        raise ValueError(
            f"总样本数错误：应为 {expected_total}，实际为 {len(merged_gdf)}"
        )

    save_gdf(merged_gdf, MERGED_OUTPUT_PATH)
    print(f"已输出：{MERGED_OUTPUT_PATH}")
    return merged_gdf


def main() -> None:
    print("开始从 tif 直接生成 2025 样本并合并四年结果\n")

    ensure_path_exists(FISHNET_PATH)
    ensure_path_exists(YEAR_SAMPLE_PATHS[2010])
    ensure_path_exists(YEAR_SAMPLE_PATHS[2015])
    ensure_path_exists(YEAR_SAMPLE_PATHS[2020])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raster_paths = discover_raster_paths()
    target_grids = select_target_grids()

    samples_2025 = generate_2025_samples(target_grids, raster_paths)
    save_gdf(samples_2025, YEAR_SAMPLE_PATHS[2025])
    print(f"已输出：{YEAR_SAMPLE_PATHS[2025]}")
    print("2025 年样本统计：")
    print(samples_2025["grid_id"].value_counts().sort_index())
    print("")

    merged_gdf = merge_all_samples()
    print("四年总样本统计：")
    print(merged_gdf["Year"].value_counts().sort_index())

    print("\n全部完成。")
    print(f"2025 样本输出：{YEAR_SAMPLE_PATHS[2025]}")
    print(f"四年总样本输出：{MERGED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

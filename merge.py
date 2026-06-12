#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import Point

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(r"D:\GisHomwork\RawData")
FISHNET_PATH = BASE_DIR / "fishnet_top62" / "fishnet_top62.shp"
MANGROVE_2025_PATH = BASE_DIR / "Mangrove_Extraction_2025_vector.shp"
INTERMEDIATE_DIR = BASE_DIR / "intermediate"
OUTPUT_DIR = BASE_DIR / "output_samples"

TARGET_FIDS = [48, 49]
SAMPLES_PER_GRID = 1000
TARGET_CRS = "EPSG:4326"

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
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        print(f"发现无效几何 {invalid_count} 个，尝试修复...")

    try:
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return gdf

    still_invalid_mask = ~gdf.geometry.is_valid
    still_invalid_count = int(still_invalid_mask.sum())
    if still_invalid_count > 0:
        print(f"仍有 {still_invalid_count} 个几何无法修复，已剔除。")
        gdf = gdf[~still_invalid_mask].copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


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


def load_2025_mangrove(target_grids: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print(f"\n读取 2025 年红树林数据：{MANGROVE_2025_PATH}")

    mangrove_meta = gpd.read_file(MANGROVE_2025_PATH, rows=1)
    mangrove_crs = mangrove_meta.crs
    if mangrove_crs is None:
        raise ValueError("2025 年红树林数据缺少坐标系定义。")

    grids_for_bbox = target_grids.to_crs(mangrove_crs)
    bbox = tuple(grids_for_bbox.total_bounds)
    print(f"按目标格网范围读取 bbox: {bbox}")

    try:
        gdf = gpd.read_file(
            MANGROVE_2025_PATH,
            bbox=bbox,
            engine="pyogrio",
            use_arrow=True,
        )
    except TypeError:
        gdf = gpd.read_file(
            MANGROVE_2025_PATH,
            bbox=bbox,
            engine="pyogrio",
        )
    except Exception:
        gdf = gpd.read_file(
            MANGROVE_2025_PATH,
            bbox=bbox,
        )

    if gdf.empty:
        raise ValueError("2025 年在目标格网范围内未读取到红树林要素。")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    print(f"2025 年 bbox 子集原始要素数：{len(gdf)}")
    print(f"2025 年 bbox 子集无效几何数：{int((~gdf.geometry.is_valid).sum())}")

    gdf = clean_polygon_gdf(gdf)
    if gdf.empty:
        raise ValueError("2025 年目标范围内要素在几何清洗后为空。")

    if "PXLVAL" not in gdf.columns:
        gdf["PXLVAL"] = 1

    gdf = gdf.to_crs(TARGET_CRS)
    print(f"2025 年 bbox 子集清洗后要素数：{len(gdf)}")
    return gdf


def clip_mangrove_by_grids(
    mangrove_gdf: gpd.GeoDataFrame,
    target_grids: gpd.GeoDataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    print(f"步骤2：裁剪 {year} 年红树林到目标格网范围")

    clipped_parts = []

    for _, grid_row in target_grids.iterrows():
        grid_id = int(grid_row["grid_id"])
        single_grid = gpd.GeoDataFrame(
            {"grid_id": [grid_id]},
            geometry=[grid_row.geometry],
            crs=target_grids.crs,
        )

        part = gpd.overlay(mangrove_gdf, single_grid, how="intersection", keep_geom_type=True)
        part = clean_polygon_gdf(part)

        if part.empty:
            raise ValueError(f"{year} 年在 grid_id={grid_id} 范围内没有红树林面，无法生成样本。")

        part["Year"] = year
        part["PXLVAL"] = 1
        clipped_parts.append(part[["grid_id", "Year", "PXLVAL", "geometry"]].copy())
        print(f"  grid_id={grid_id} 裁剪后面数：{len(part)}")

    clipped = gpd.GeoDataFrame(pd.concat(clipped_parts, ignore_index=True), crs=TARGET_CRS)
    return clipped


def random_points_from_polygons(
    grid_polygons: gpd.GeoDataFrame,
    n_points: int,
) -> list:
    valid_polygons = grid_polygons[grid_polygons.geometry.notna()].copy()
    valid_polygons = valid_polygons[~valid_polygons.geometry.is_empty].copy()

    if valid_polygons.empty:
        raise ValueError("没有可用于采样的红树林面。")

    area_gdf = valid_polygons.to_crs("EPSG:6933")
    weights = area_gdf.geometry.area.to_numpy(dtype=float)

    if weights.size == 0 or weights.sum() <= 0:
        weights = np.ones(len(valid_polygons), dtype=float)

    probabilities = weights / weights.sum()
    source_geometries = valid_polygons.geometry.tolist()

    points = []
    attempts = 0
    max_attempts = max(100000, n_points * 2000)

    while len(points) < n_points and attempts < max_attempts:
        polygon_index = int(np.random.choice(len(source_geometries), p=probabilities))
        geometry = source_geometries[polygon_index]

        if not geometry.is_valid:
            geometry = make_valid(geometry)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty:
            attempts += 1
            continue

        minx, miny, maxx, maxy = geometry.bounds
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )

        if geometry.covers(point):
            points.append(point)

        attempts += 1

    if len(points) != n_points:
        raise RuntimeError(
            f"随机撒点失败：目标 {n_points} 个，实际仅生成 {len(points)} 个。"
        )

    return points



def generate_samples_for_year(
    clipped_mangrove: gpd.GeoDataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    print(f"步骤3-4：生成 {year} 年样本点")

    year_samples = []

    for grid_id in TARGET_FIDS:
        grid_polygons = clipped_mangrove[clipped_mangrove["grid_id"] == grid_id].copy()

        if grid_polygons.empty:
            raise ValueError(f"{year} 年 grid_id={grid_id} 没有红树林面，无法生成样本。")

        print(f"  grid_id={grid_id} 参与采样的面数量：{len(grid_polygons)}")
        points = random_points_from_polygons(grid_polygons, SAMPLES_PER_GRID)

        sample_gdf = gpd.GeoDataFrame(
            {
                "grid_id": [grid_id] * SAMPLES_PER_GRID,
                "Year": [year] * SAMPLES_PER_GRID,
                "PXLVAL": [1] * SAMPLES_PER_GRID,
            },
            geometry=points,
            crs=TARGET_CRS,
        )
        year_samples.append(sample_gdf)
        print(f"  grid_id={grid_id} 已生成 {len(sample_gdf)} 个点")

    samples = gpd.GeoDataFrame(pd.concat(year_samples, ignore_index=True), crs=TARGET_CRS)
    validate_samples(samples, year)
    return samples



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


def merge_all_samples() -> gpd.GeoDataFrame:
    print("步骤5：合并 2010/2015/2020/2025 四年样本")

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
    print("开始生成 2025 年样本并合并四年结果\n")

    ensure_path_exists(FISHNET_PATH)
    ensure_path_exists(MANGROVE_2025_PATH)
    ensure_path_exists(YEAR_SAMPLE_PATHS[2010])
    ensure_path_exists(YEAR_SAMPLE_PATHS[2015])
    ensure_path_exists(YEAR_SAMPLE_PATHS[2020])

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_grids = select_target_grids()
    save_gdf(target_grids, INTERMEDIATE_DIR / "target_grids.shp")
    print(f"已输出：{INTERMEDIATE_DIR / 'target_grids.shp'}")

    mangrove_2025 = load_2025_mangrove(target_grids)
    clipped_2025 = clip_mangrove_by_grids(mangrove_2025, target_grids, 2025)

    clipped_path = INTERMEDIATE_DIR / "mangrove_2025_in_grids.shp"
    save_gdf(clipped_2025, clipped_path)
    print(f"已输出：{clipped_path}")

    samples_2025 = generate_samples_for_year(clipped_2025, 2025)
    save_gdf(samples_2025, YEAR_SAMPLE_PATHS[2025])
    print(f"已输出：{YEAR_SAMPLE_PATHS[2025]}")
    print("2025 年样本统计：")
    print(samples_2025["grid_id"].value_counts().sort_index())
    print("")

    merged_gdf = merge_all_samples()
    print("四年总样本统计：")
    print(merged_gdf["Year"].value_counts().sort_index())

    print("\n全部完成。")
    print(f"2025 样本输出目录：{YEAR_SAMPLE_PATHS[2025]}")
    print(f"四年总样本输出目录：{MERGED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

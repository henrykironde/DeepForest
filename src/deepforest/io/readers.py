import logging

import pandas as pd
from shapely.geometry import box, shape

from ..models.annotation import Annotation, AnnotationCollection
from .path_utils import resolve_image_path

logger = logging.getLogger(__name__)

def _dataframe_from_csv(path: str) -> pd.DataFrame:
    # minimal CSV reader — users can extend for custom columns
    return pd.read_csv(path)

def read_file(input_data: str | pd.DataFrame, image_path: str | None = None, root_dir: str | None = None, *, validate: bool = True) -> AnnotationCollection:
    """
    Read annotations from input_data and return an AnnotationCollection with canonical fields:
      - image_path (resolved if possible)
      - label
      - geometry (shapely geometry in image pixel coordinates)

    input_data:
      - str path to a CSV (or other format dispatch can be added)
      - pandas.DataFrame already loaded

    image_path:
      - If provided, used to fill missing image_path columns.
      - Can be absolute or relative; resolve rules applied.

    root_dir:
      - optional directory used to resolve relative image paths.

    Note: This function is intended to be a canonical normalizer. It does not attempt to change old read_file APIs.
    """
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        if str(input_data).lower().endswith(".csv"):
            df = _dataframe_from_csv(input_data)
        else:
            # could add shapefile/geojson readers here (geopandas optional)
            raise NotImplementedError("Only CSV and DataFrame inputs implemented in this minimal reader")

    # fill image_path if missing
    if "image_path" not in df.columns and image_path:
        df["image_path"] = image_path

    anns = AnnotationCollection()
    for _, row in df.iterrows():
        raw_imgp = row.get("image_path")
        resolved_imgp, inferred_root = resolve_image_path(raw_imgp or image_path, root_dir)
        raw_geom = row.get("geometry")

        geom = raw_geom
        # if geometry is provided as dict-like (GeoJSON), convert
        if isinstance(raw_geom, dict):
            geom = shape(raw_geom)
        # if geometry is provided as xmin,ymin,xmax,ymax columns
        elif raw_geom is None and {"xmin","ymin","xmax","ymax"}.issubset(df.columns):
            xmin = row["xmin"]
            ymin = row["ymin"]
            xmax = row["xmax"]
            ymax = row["ymax"]
            geom = box(xmin, ymin, xmax, ymax)

        ann = Annotation(
            image_path=resolved_imgp,
            label=row.get("label"),
            geometry=geom,
            properties={k: v for k, v in row.items() if k not in ("image_path", "label", "geometry")}
        )
        anns.append(ann)

    if validate:
        anns.validate()
    return anns

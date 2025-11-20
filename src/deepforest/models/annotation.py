from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry


@dataclass
class Annotation:
    image_path: str
    label: str
    geometry: BaseGeometry
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"image_path": self.image_path, "label": self.label, "geometry": self.geometry}
        d.update(self.properties)
        return d

class AnnotationCollection:
    """
    Lightweight container for annotations with helpers to convert to/from pandas/GeoDataFrame.
    Geometry objects are expected to be shapely geometries in image coordinates (pixels).
    """
    def __init__(self, annotations: Iterable[Annotation] | None = None):
        self.annotations: list[Annotation] = list(annotations) if annotations else []

    def append(self, ann: Annotation) -> None:
        self.annotations.append(ann)

    def extend(self, anns: Iterable[Annotation]) -> None:
        self.annotations.extend(anns)

    def to_dataframe(self):
        import pandas as pd
        rows = [a.to_dict() for a in self.annotations]
        df = pd.DataFrame(rows)
        # Keep shapely geometries in 'geometry' column
        return df

    def validate(self) -> bool:
        from shapely.geometry.base import BaseGeometry as ShapelyBase
        for i, a in enumerate(self.annotations):
            if not a.image_path:
                raise ValueError(f"annotation {i} missing image_path")
            if a.label is None:
                raise ValueError(f"annotation {i} missing label")
            if not isinstance(a.geometry, ShapelyBase):
                raise ValueError(f"annotation {i} geometry is not a shapely geometry")
        return True

    @classmethod
    def from_dataframe(cls, df):
        from shapely import wkt
        anns = []
        for _, row in df.iterrows():
            geom = row.get("geometry")
            # support WKT strings
            if isinstance(geom, str):
                geom = wkt.loads(geom)
            ann = Annotation(
                image_path=row["image_path"],
                label=row.get("label"),
                geometry=geom,
                properties={k: v for k, v in row.items() if k not in ("image_path", "label", "geometry")}
            )
            anns.append(ann)
        return cls(anns)

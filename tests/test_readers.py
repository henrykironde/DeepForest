import pandas as pd
from shapely.geometry import Point

from deepforest.io.readers import read_file
from deepforest.models.annotation import AnnotationCollection


def test_read_dataframe_point():
    df = pd.DataFrame([{"image_path":"images/img_1.png", "label":"tree", "geometry": Point(10,20)}])
    anns = read_file(df)
    assert isinstance(anns, AnnotationCollection)
    assert len(anns.annotations) == 1
    assert anns.annotations[0].label == "tree"


def test_read_dataframe_bbox_xyxy():
    df = pd.DataFrame([{"image_path":"images/img_2.png", "label":"bush", "xmin":1, "ymin":2, "xmax":10, "ymax":12}])
    anns = read_file(df)
    assert len(anns.annotations) == 1
    assert anns.annotations[0].geometry.bounds == (1.0,2.0,10.0,12.0)

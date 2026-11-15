from computer_vision.risk import assess, classify, risk_score
from computer_vision.zones import Zone, foot_point, load_zones, point_in_polygon, zones_for_box

SQUARE = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]


def test_point_in_polygon_inside_outside_edge():
    assert point_in_polygon((0.75, 0.75), SQUARE)
    assert not point_in_polygon((0.25, 0.25), SQUARE)
    assert point_in_polygon((0.5, 0.75), SQUARE)  # on edge counts as inside


def test_point_in_concave_polygon():
    l_shape = [(0, 0), (1, 0), (1, 0.4), (0.4, 0.4), (0.4, 1), (0, 1)]
    assert point_in_polygon((0.2, 0.8), l_shape)
    assert not point_in_polygon((0.8, 0.8), l_shape)


def test_foot_point_and_zone_lookup():
    assert foot_point((10, 20, 30, 100)) == (20.0, 100)
    zones = [Zone("hazard", SQUARE, ["*"]), Zone("cam2_only", SQUARE, ["cam_2"])]
    # 200x200 image, person standing at (150, 180) -> normalised (0.75, 0.9)
    assert zones_for_box((140, 50, 160, 180), zones, "cam_1", 200, 200) == ["hazard"]
    assert zones_for_box((140, 50, 160, 180), zones, "cam_2", 200, 200) == ["hazard", "cam2_only"]
    assert zones_for_box((10, 10, 30, 60), zones, "cam_1", 200, 200) == []


def test_load_zones_from_config(cfg):
    zones = load_zones(cfg)
    assert zones and all(len(z.polygon) >= 3 for z in zones)


def test_risk_scoring_and_levels(cfg):
    assert risk_score([], cfg) == 0 and classify(0, cfg) == "None"
    assert assess(["missing_gloves"], cfg) == (1, "Low")
    assert assess(["missing_helmet"], cfg) == (3, "Medium")
    assert assess(["missing_helmet", "missing_vest"], cfg) == (5, "Medium")
    assert assess(["missing_helmet", "restricted_zone"], cfg) == (7, "High")
    assert assess(["forklift_proximity", "missing_helmet"], cfg) == (6, "High")

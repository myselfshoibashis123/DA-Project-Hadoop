from computer_vision.detector import Detection, iou, merge_person_detections
from computer_vision.safety_analysis import analyse_frame, associate, near_forklift, overlap_ratio, ppe_status
from computer_vision.zones import Zone


def D(label, box, conf=0.9):
    return Detection(label=label, confidence=conf, bbox=box)


def test_iou_and_overlap():
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert overlap_ratio((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0
    assert abs(overlap_ratio((5, 0, 15, 10), (0, 0, 10, 10)) - 0.5) < 1e-9


def test_merge_person_detections_removes_duplicates():
    merged = merge_person_detections([D("Person", (0, 0, 10, 20), 0.9), D("Person", (1, 0, 10, 20), 0.6),
                                      D("Person", (50, 0, 60, 20), 0.7)], 0.5)
    assert len(merged) == 2 and merged[0].confidence == 0.9


def test_merge_drops_box_spanning_two_workers():
    a, b = D("Person", (2, 62, 216, 611), 0.93), D("Person", (208, 82, 330, 557), 0.92)
    spanning = D("Person", (81, 71, 327, 507), 0.48)
    occluded = D("Person", (180, 60, 300, 400), 0.40)  # mostly visible worker behind 'a'
    kept = merge_person_detections([a, b, spanning], 0.5, 0.7)
    assert spanning not in kept and len(kept) == 2
    c = D("Person", (500, 50, 600, 400), 0.9)
    assert occluded in merge_person_detections([c, a, occluded], 0.5, 0.7)


def test_association_picks_containing_person():
    persons = [D("Person", (0, 0, 100, 300)), D("Person", (200, 0, 300, 300))]
    objs = [D("helmet", (20, 0, 60, 40)), D("no_vest", (210, 80, 290, 180)), D("helmet", (500, 0, 540, 40))]
    mapping, unassigned = associate(persons, objs, 0.5)
    assert [o.label for o in mapping[0]] == ["helmet"]
    assert [o.label for o in mapping[1]] == ["no_vest"]
    assert [o.bbox for o in unassigned] == [(500, 0, 540, 40)]


def test_ppe_status_explicit_negative_wins_by_confidence(cfg):
    st = ppe_status([D("no_helmet", (0, 0, 1, 1), 0.8), D("helmet", (0, 0, 1, 1), 0.4), D("vest", (0, 0, 1, 1))], cfg)
    assert st["helmet"]["status"] == "missing"
    assert st["vest"]["status"] == "worn"
    assert st["gloves"]["status"] == "unknown"


def test_unknown_as_missing_policy(cfg):
    cfg["ppe_policy"]["treat_unknown_as_missing"] = ["gloves"]
    st = ppe_status([], cfg)
    assert st["gloves"] == {"status": "missing", "confidence": None, "inferred": True}


def test_forklift_proximity_and_operator_exclusion():
    person = D("Person", (100, 100, 150, 300))           # height 200, foot point (125, 300)
    near = D("forklift", (160, 150, 400, 320))           # gap 35 px -> 0.175 h
    far = D("forklift", (600, 100, 900, 300))
    assert near_forklift(person, [far], 0.75) is None
    assert abs(near_forklift(person, [near, far], 0.75) - 0.175) < 1e-6
    operator = D("Person", (200, 160, 260, 270))         # inside forklift box, feet above its ground line
    assert near_forklift(operator, [near], 0.75) is None
    pedestrian = D("Person", (200, 160, 260, 330))       # overlaps forklift box but stands on the ground
    assert abs(near_forklift(pedestrian, [near], 0.75) - 10 / 170) < 1e-6  # flagged, 10 px gap


def test_analyse_frame_end_to_end(cfg):
    cfg["zones"] = []
    zones = [Zone("hazard", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)], ["*"])]
    persons = [D("Person", (10, 10, 110, 390)), D("Person", (300, 10, 380, 390))]
    objects = [
        D("helmet", (30, 10, 80, 60)), D("vest", (20, 100, 100, 200)),        # worker 0 compliant
        D("no_helmet", (310, 10, 360, 60)), D("vest", (305, 100, 375, 200)),  # worker 1 no helmet, in zone
        D("no_gloves", (385, 300, 398, 320)),   # unassigned violation (no person box), inside zone
    ]
    res = analyse_frame(persons, objects, cfg, zones, "cam_1", 400, 400)
    w0, w1, u0 = res
    assert w0["violations"] == [] and w0["risk_level"] == "None"
    assert w1["violations"] == ["missing_helmet", "restricted_zone"]
    assert (w1["risk_score"], w1["risk_level"]) == (7, "High")
    assert w1["zones"] == ["hazard"]
    assert u0["assigned"] is False and u0["violations"] == ["missing_gloves", "restricted_zone"]

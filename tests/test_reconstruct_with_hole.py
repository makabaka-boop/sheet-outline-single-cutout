"""End-to-end tests for the outer-contour-plus-single-hole endpoint.

The placement verdicts are checked against two oracles implemented here
from scratch, independently of app.geometry:

* ``oracle_point_in_polygon`` casts a *vertical* ray over the raw,
  unordered edge set (the app casts a horizontal ray over ordered
  vertices);
* ``oracle_segments_intersect`` is a general cross-product-sign test
  (the app uses axis-aligned interval logic).
"""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def seg(seg_id, x1, y1, x2, y2):
    return {"id": seg_id, "a": [x1, y1], "b": [x2, y2]}


def rect(prefix, x1, y1, x2, y2):
    """Axis-aligned rectangle as four edges, counter-clockwise."""
    return [
        seg(f"{prefix}1", x1, y1, x2, y1),
        seg(f"{prefix}2", x2, y1, x2, y2),
        seg(f"{prefix}3", x2, y2, x1, y2),
        seg(f"{prefix}4", x1, y2, x1, y1),
    ]


def post(outer, hole):
    return client.post(
        "/reconstruct_with_hole",
        json={"outer": {"segments": outer}, "hole": {"segments": hole}},
    )


# --- independent oracles ------------------------------------------------------


def oracle_point_in_polygon(point, edges):
    """Even-odd test with a +y ray over an unordered edge list."""
    x, y = point
    inside = False
    for s in edges:
        (x1, y1), (x2, y2) = s["a"], s["b"]
        if x1 == x2:
            continue  # vertical edge: parallel to the ray
        if y1 > y and min(x1, x2) <= x < max(x1, x2):
            inside = not inside
    return inside


def oracle_segments_intersect(p1, p2, p3, p4):
    """Closed-segment intersection via cross-product signs (general case)."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    def on_segment(a, b, p):
        return (
            cross(a, b, p) == 0
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

    return (
        (d1 == 0 and on_segment(p3, p4, p1))
        or (d2 == 0 and on_segment(p3, p4, p2))
        or (d3 == 0 and on_segment(p1, p2, p3))
        or (d4 == 0 and on_segment(p1, p2, p4))
    )


def oracle_contact_witness(outer, hole):
    """Smallest (id, id) pair among intersecting outer/hole edge pairs."""
    pairs = []
    for s in outer:
        for t in hole:
            if oracle_segments_intersect(
                tuple(s["a"]), tuple(s["b"]), tuple(t["a"]), tuple(t["b"])
            ):
                pairs.append(tuple(sorted((s["id"], t["id"]))))
    if not pairs:
        return None
    first, second = min(pairs)
    return {"segment_id": first, "other_segment_id": second}


def oracle_placement(outer, hole):
    """'contact' / 'outside' / 'inside' derived purely from the oracles."""
    if oracle_contact_witness(outer, hole) is not None:
        return "contact"
    hole_vertices = {tuple(s["a"]) for s in hole} | {tuple(s["b"]) for s in hole}
    if all(oracle_point_in_polygon(p, outer) for p in hole_vertices):
        return "inside"
    return "outside"


# --- response verification ------------------------------------------------------


def verify_ring(ring, segments, *, clockwise):
    """Re-derive every guarantee of one ring in a 200 response."""
    vertices = [tuple(v) for v in ring["vertices"]]
    edge_ids = ring["edge_ids"]
    by_id = {s["id"]: s for s in segments}
    n = len(vertices)

    assert n == len(segments) == len(edge_ids)
    assert set(edge_ids) == set(by_id)  # every edge used exactly once
    assert vertices[0] == min(vertices)  # lexicographically smallest start

    for i in range(n):  # edge i connects vertices[i] to vertices[i+1]
        s = by_id[edge_ids[i]]
        assert {tuple(s["a"]), tuple(s["b"])} == {vertices[i], vertices[(i + 1) % n]}

    signed = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    if clockwise:
        assert signed < 0  # outer ring: clockwise
    else:
        assert signed > 0  # hole ring: counter-clockwise
    assert ring["doubled_area"] == abs(signed)

    perimeter = sum(
        abs(s["a"][0] - s["b"][0]) + abs(s["a"][1] - s["b"][1]) for s in segments
    )
    assert ring["perimeter"] == perimeter


def verify_success(body, outer, hole):
    verify_ring(body["outer"], outer, clockwise=True)
    verify_ring(body["hole"], hole, clockwise=False)
    assert (
        body["net_doubled_area"]
        == body["outer"]["doubled_area"] - body["hole"]["doubled_area"]
    )
    assert (
        body["total_cut_length"]
        == body["outer"]["perimeter"] + body["hole"]["perimeter"]
    )
    # independent placement proof on the accepted response
    assert oracle_placement(outer, hole) == "inside"


# --- fixtures ---------------------------------------------------------------

# Concave L-shaped outer contour, reflex vertex at (4, 4); the bay
# x in (4, 8), y in (4, 8) is outside the sheet but inside its bounding box.
OUTER_L = [
    seg("o1", 0, 0, 8, 0),
    seg("o2", 8, 0, 8, 4),
    seg("o3", 8, 4, 4, 4),
    seg("o4", 4, 4, 4, 8),
    seg("o5", 4, 8, 0, 8),
    seg("o6", 0, 8, 0, 0),
]

HOLE_BOTTOM = rect("h", 1, 1, 3, 3)  # strictly inside the lower limb
HOLE_LEFT = rect("h", 1, 5, 3, 7)  # strictly inside the left limb
HOLE_IN_BAY = rect("h", 5, 5, 7, 7)  # inside the bounding box, in the bay
HOLE_STRADDLE = rect("h", 3, 5, 5, 7)  # half in the left limb, half in the bay
HOLE_POINT_TOUCH = rect("h", 2, 2, 4, 4)  # one corner on the reflex vertex (4, 4)
HOLE_SHARED_EDGE = rect("h", 1, 0, 3, 2)  # bottom edge collinear with o1
HOLE_FAR = rect("h", 20, 20, 22, 22)  # completely outside


def shuffled_pairs(outer, hole, rounds=5):
    """Deterministic independent shuffles of both segment groups."""
    orders = [(list(outer), list(hole))]
    rng = random.Random(20260926)
    for _ in range(rounds):
        o, h = list(outer), list(hole)
        rng.shuffle(o)
        rng.shuffle(h)
        orders.append((o, h))
    return orders


# --- happy paths -------------------------------------------------------------


def test_hole_inside_concave_outer_exact_response():
    resp = post(OUTER_L, HOLE_BOTTOM)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "outer": {
            "vertices": [[0, 0], [0, 8], [4, 8], [4, 4], [8, 4], [8, 0]],
            "edge_ids": ["o6", "o5", "o4", "o3", "o2", "o1"],
            "doubled_area": 96,
            "perimeter": 32,
        },
        "hole": {
            "vertices": [[1, 1], [3, 1], [3, 3], [1, 3]],
            "edge_ids": ["h1", "h2", "h3", "h4"],
            "doubled_area": 8,
            "perimeter": 8,
        },
        "net_doubled_area": 88,
        "total_cut_length": 40,
    }
    verify_success(body, OUTER_L, HOLE_BOTTOM)


def test_hole_inside_other_limb_of_concave_outer():
    resp = post(OUTER_L, HOLE_LEFT)
    assert resp.status_code == 200
    verify_success(resp.json(), OUTER_L, HOLE_LEFT)


def test_big_integer_areas():
    m = 10**6
    outer = rect("o", -m, -m, m, m)
    hole = rect("h", -(m - 1), -(m - 1), m - 1, m - 1)
    resp = post(outer, hole)
    assert resp.status_code == 200
    body = resp.json()
    outer_doubled = 2 * (2 * m) ** 2
    hole_doubled = 2 * (2 * (m - 1)) ** 2
    assert body["outer"]["doubled_area"] == outer_doubled
    assert body["hole"]["doubled_area"] == hole_doubled
    assert body["net_doubled_area"] == outer_doubled - hole_doubled
    assert body["total_cut_length"] == 8 * m + 8 * (m - 1)
    verify_success(body, outer, hole)


def test_success_is_shuffle_stable():
    responses = [post(o, h).json() for o, h in shuffled_pairs(OUTER_L, HOLE_BOTTOM)]
    assert all(r == responses[0] for r in responses[1:])


# --- placement rejections ------------------------------------------------------


def test_hole_in_concave_bay_is_outside():
    # The hole sits inside the outer bounding box; only a real
    # point-in-polygon test can see that it lies in the bay.
    assert oracle_placement(OUTER_L, HOLE_IN_BAY) == "outside"
    resp = post(OUTER_L, HOLE_IN_BAY)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_OUTSIDE"
    assert detail["witness"] == {"segment_id": "h1"}


def test_hole_straddling_concave_notch_needs_more_than_one_vertex():
    # Vertices (3, 5) and (3, 7) are inside the outer ring, (5, 5) and
    # (5, 7) are in the bay: probing a single hole vertex cannot decide.
    inside = [oracle_point_in_polygon(tuple(s["a"]), OUTER_L) for s in HOLE_STRADDLE]
    assert any(inside) and not all(inside)
    resp = post(OUTER_L, HOLE_STRADDLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_BOUNDARY_CONTACT"
    assert detail["witness"] == oracle_contact_witness(OUTER_L, HOLE_STRADDLE)


def test_single_point_touch_is_rejected():
    # The hole only shares the reflex vertex (4, 4) with the outer ring.
    assert oracle_placement(OUTER_L, HOLE_POINT_TOUCH) == "contact"
    resp = post(OUTER_L, HOLE_POINT_TOUCH)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_BOUNDARY_CONTACT"
    assert detail["witness"] == oracle_contact_witness(OUTER_L, HOLE_POINT_TOUCH)


def test_shared_collinear_edge_is_rejected():
    assert oracle_placement(OUTER_L, HOLE_SHARED_EDGE) == "contact"
    resp = post(OUTER_L, HOLE_SHARED_EDGE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_BOUNDARY_CONTACT"
    assert detail["witness"] == oracle_contact_witness(OUTER_L, HOLE_SHARED_EDGE)


def test_hole_far_outside():
    resp = post(OUTER_L, HOLE_FAR)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_OUTSIDE"
    assert detail["witness"] == {"segment_id": "h1"}


def test_rejection_is_shuffle_stable():
    for hole in (HOLE_IN_BAY, HOLE_STRADDLE, HOLE_POINT_TOUCH, HOLE_SHARED_EDGE):
        details = [post(o, h).json() for o, h in shuffled_pairs(OUTER_L, hole)]
        assert all(d == details[0] for d in details[1:])


def test_random_rectangles_match_oracle():
    rng = random.Random(20260926)
    for _ in range(60):
        ox1, ox2 = sorted(rng.sample(range(-40, 40), 2))
        oy1, oy2 = sorted(rng.sample(range(-40, 40), 2))
        outer = rect("o", ox1, oy1, ox2, oy2)
        mode = rng.choice(["inside", "outside", "any"])
        if mode == "inside" and ox2 - ox1 > 2 and oy2 - oy1 > 2:
            hx1 = rng.randint(ox1 + 1, ox2 - 2)
            hx2 = rng.randint(hx1 + 1, ox2 - 1)
            hy1 = rng.randint(oy1 + 1, oy2 - 2)
            hy2 = rng.randint(hy1 + 1, oy2 - 1)
        elif mode == "outside":
            hx1 = ox2 + rng.randint(1, 10)
            hx2 = hx1 + rng.randint(1, 5)
            hy1 = rng.randint(-40, 35)
            hy2 = hy1 + rng.randint(1, 5)
        else:
            hx1, hx2 = sorted(rng.sample(range(-45, 45), 2))
            hy1, hy2 = sorted(rng.sample(range(-45, 45), 2))
        hole = rect("h", hx1, hy1, hx2, hy2)

        resp = post(outer, hole)
        placement = oracle_placement(outer, hole)
        if placement == "inside":
            assert resp.status_code == 200
            verify_success(resp.json(), outer, hole)
        elif placement == "contact":
            assert resp.status_code == 422
            detail = resp.json()["detail"]
            assert detail["code"] == "HOLE_BOUNDARY_CONTACT"
            assert detail["witness"] == oracle_contact_witness(outer, hole)
        else:
            assert resp.status_code == 422
            detail = resp.json()["detail"]
            assert detail["code"] == "HOLE_OUTSIDE"
            assert detail["witness"] == {"segment_id": "h1"}


# --- per-group validation ------------------------------------------------------


def test_total_segment_count_bound():
    outer = [seg(f"b{i}", i, 0, i + 1, 0) for i in range(249)]
    outer += [seg(f"t{i}", i, 4, i + 1, 4) for i in range(249)]
    outer += [seg("v0", 0, 0, 0, 4), seg("v1", 249, 0, 249, 4)]
    assert len(outer) == 500
    resp = post(outer, HOLE_BOTTOM)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INVALID_SEGMENT_COUNT"
    assert detail["witness"] == {"count": 504}


def test_group_segment_count_bound():
    resp = post(OUTER_L, HOLE_BOTTOM[:3])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"


def test_duplicate_id_across_groups():
    hole = rect("o", 1, 1, 3, 3)  # ids "o1".."o4" collide with the outer group
    resp = post(OUTER_L, hole)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DUPLICATE_SEGMENT_ID"
    assert detail["witness"] == {"segment_id": "o1"}


def test_hole_group_uses_single_ring_checks():
    bad_hole = HOLE_BOTTOM + [seg("h5", 2, 3, 2, 5)]  # dangling edge off h3
    resp = post(OUTER_L, bad_hole)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "BAD_DEGREE"
    assert detail["witness"] == {"point": [2, 3]}

    tilted = HOLE_BOTTOM[:3] + [seg("h4", 1, 3, 2, 2)]
    resp = post(OUTER_L, tilted)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "NON_AXIS_ALIGNED"


def test_outer_group_is_validated_first():
    bad_outer = OUTER_L[:5] + [seg("o6", 0, 8, 1, 1)]  # non-axis-aligned
    bad_hole = HOLE_BOTTOM[:3] + [seg("h4", 1, 3, 1, 1)]  # zero length
    resp = post(bad_outer, bad_hole)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "NON_AXIS_ALIGNED"


def test_malformed_hole_payload():
    resp = client.post("/reconstruct_with_hole", json={"outer": {"segments": OUTER_L}})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

    resp = client.post("/reconstruct_with_hole", json=[1, 2, 3])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

    resp = client.post(
        "/reconstruct_with_hole",
        json={"outer": {"edges": []}, "hole": {"segments": HOLE_BOTTOM}},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

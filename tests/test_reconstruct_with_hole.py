"""End-to-end tests for the outer+hole reconstruction entry point.

Placement guarantees are verified with two oracles that are independent
of the implementation: a winding-number point-in-polygon test and a
general orientation-based segment-intersection test (CLRS), while the
implementation relies on axis-aligned ray casting and specialised H/V
predicates.
"""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def seg(seg_id, x1, y1, x2, y2):
    return {"id": seg_id, "a": [x1, y1], "b": [x2, y2]}


def post(outer, hole):
    return client.post("/reconstruct-with-hole", json={"outer": outer, "hole": hole})


# --- independent oracles ------------------------------------------------------


def _cross(a, b, c):
    """Z-component of (b - a) x (c - a) (a.k.a. isLeft for edge a -> b)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def oracle_segments_intersect(p1, p2, p3, p4):
    """True iff the two closed segments share at least one point (CLRS)."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    def on_segment(pi, pj, pk):  # pk collinear with pi-pj: is it between them?
        return min(pi[0], pj[0]) <= pk[0] <= max(pi[0], pj[0]) and min(
            pi[1], pj[1]
        ) <= pk[1] <= max(pi[1], pj[1])

    return (
        (d1 == 0 and on_segment(p3, p4, p1))
        or (d2 == 0 and on_segment(p3, p4, p2))
        or (d3 == 0 and on_segment(p1, p2, p3))
        or (d4 == 0 and on_segment(p1, p2, p4))
    )


def oracle_point_inside(point, vertices):
    """Non-zero winding number.  Never queried with boundary points."""
    x, y = point
    winding = 0
    n = len(vertices)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        if a[1] <= y:
            if b[1] > y and _cross(a, b, point) > 0:
                winding += 1
        elif b[1] <= y and _cross(a, b, point) < 0:
            winding -= 1
    return winding != 0


# --- independent recomputation -------------------------------------------------


def verify_loop(loop_body, raw_segments, want_clockwise):
    """Re-derive every per-loop guarantee of a 200 response from raw input."""
    vertices = [tuple(v) for v in loop_body["vertices"]]
    edge_ids = loop_body["edge_ids"]
    by_id = {s["id"]: s for s in raw_segments}
    n = len(vertices)

    assert n == len(raw_segments) == len(edge_ids)
    assert set(edge_ids) == set(by_id)  # every edge used exactly once
    assert vertices[0] == min(vertices)  # lexicographically smallest start

    # edge i must connect vertices[i] to vertices[i+1]
    for i in range(n):
        s = by_id[edge_ids[i]]
        assert {tuple(s["a"]), tuple(s["b"])} == {vertices[i], vertices[(i + 1) % n]}

    # integer shoelace, recomputed from scratch
    signed = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    assert (signed < 0) == want_clockwise  # outer clockwise, hole counter-clockwise
    assert loop_body["doubled_area"] == abs(signed)
    assert loop_body["perimeter"] == sum(
        abs(s["a"][0] - s["b"][0]) + abs(s["a"][1] - s["b"][1]) for s in raw_segments
    )
    return vertices


def verify_hole_strictly_inside(outer_segments, hole_segments, outer_vertices):
    """Strict containment, fully re-derived with the independent oracles."""
    for o in outer_segments:
        for h in hole_segments:
            assert not oracle_segments_intersect(
                tuple(o["a"]), tuple(o["b"]), tuple(h["a"]), tuple(h["b"])
            )
    for h in hole_segments:
        assert oracle_point_inside(tuple(h["a"]), outer_vertices)


def expected_contact_witness(outer_segments, hole_segments):
    """Smallest edge id involved in any boundary contact, per the oracle."""
    involved = set()
    for o in outer_segments:
        for h in hole_segments:
            if oracle_segments_intersect(
                tuple(o["a"]), tuple(o["b"]), tuple(h["a"]), tuple(h["b"])
            ):
                involved.add(o["id"])
                involved.add(h["id"])
    assert involved, "fixture must actually have boundary contact"
    return min(involved)


def assert_no_contact(outer_segments, hole_segments):
    for o in outer_segments:
        for h in hole_segments:
            assert not oracle_segments_intersect(
                tuple(o["a"]), tuple(o["b"]), tuple(h["a"]), tuple(h["b"])
            )


def shuffles(outer, hole, rounds=5):
    """Deterministic independent permutations of the two groups."""
    pairs = [(list(outer), list(hole)), (list(reversed(outer)), list(reversed(hole)))]
    rng = random.Random(20260926)
    for _ in range(rounds):
        o = list(outer)
        h = list(hole)
        rng.shuffle(o)
        rng.shuffle(h)
        pairs.append((o, h))
    return pairs


# --- fixtures ------------------------------------------------------------------

# 10x10 outer square; ids deliberately shuffled.
OUTER = [
    seg("o3", 10, 10, 0, 10),
    seg("o1", 0, 0, 10, 0),
    seg("o4", 0, 10, 0, 0),
    seg("o2", 10, 0, 10, 10),
]

# 2x2 hole strictly inside OUTER.
HOLE = [
    seg("h2", 5, 3, 5, 5),
    seg("h4", 3, 5, 3, 3),
    seg("h1", 3, 3, 5, 3),
    seg("h3", 5, 5, 3, 5),
]

# Concave L-shaped outer: solid [0,8]x[0,4] u [0,4]x[4,8], reflex at (4,4).
L_OUTER = [
    seg("L2", 8, 0, 8, 4),
    seg("L5", 4, 8, 0, 8),
    seg("L1", 0, 0, 8, 0),
    seg("L4", 4, 4, 4, 8),
    seg("L6", 0, 8, 0, 0),
    seg("L3", 8, 4, 4, 4),
]
# Vertex loop of L_OUTER (any consistent order works for the oracles).
L_POLYGON = [(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)]

# Hole inside the bottom-right arm of the L (beyond the reflex vertex).
L_HOLE = [
    seg("p1", 5, 1, 7, 1),
    seg("p2", 7, 1, 7, 3),
    seg("p3", 7, 3, 5, 3),
    seg("p4", 5, 3, 5, 1),
]

# Hole whose lexicographically smallest vertex (3,3) lies inside the L, but
# whose top-right corner sticks into the notch: edges cross the L boundary.
L_STRADDLE_HOLE = [
    seg("q1", 3, 3, 6, 3),
    seg("q2", 6, 3, 6, 6),
    seg("q3", 6, 6, 3, 6),
    seg("q4", 3, 6, 3, 3),
]

# Hole inside the bounding box but in the notch of the L: outside the solid.
L_NOTCH_HOLE = [
    seg("n1", 5, 5, 7, 5),
    seg("n2", 7, 5, 7, 7),
    seg("n3", 7, 7, 5, 7),
    seg("n4", 5, 7, 5, 5),
]

# Hole straddling the outer right edge: two clean crossings at (10,2),(10,6).
CROSS_HOLE = [
    seg("x1", 8, 2, 12, 2),
    seg("x2", 12, 2, 12, 6),
    seg("x3", 12, 6, 8, 6),
    seg("x4", 8, 6, 8, 2),
]

# Hole sharing a collinear interval [2,6] with the outer bottom edge.
SHARED_EDGE_HOLE = [
    seg("s1", 2, 0, 6, 0),
    seg("s2", 6, 0, 6, 4),
    seg("s3", 6, 4, 2, 4),
    seg("s4", 2, 4, 2, 0),
]

# Hole touching the outer contour at the single corner point (10,10).
POINT_TOUCH_HOLE = [
    seg("t1", 10, 10, 14, 10),
    seg("t2", 14, 10, 14, 14),
    seg("t3", 14, 14, 10, 14),
    seg("t4", 10, 14, 10, 10),
]

# Hole fully outside, no contact at all.
OUTSIDE_HOLE = [
    seg("w1", 20, 20, 24, 20),
    seg("w2", 24, 20, 24, 24),
    seg("w3", 24, 24, 20, 24),
    seg("w4", 20, 24, 20, 20),
]

# Valid loop that surrounds the outer contour instead of sitting inside.
SURROUNDING_HOLE = [
    seg("r1", -5, -5, 15, -5),
    seg("r2", 15, -5, 15, 15),
    seg("r3", 15, 15, -5, 15),
    seg("r4", -5, 15, -5, -5),
]

# Same geometry as the outer loop, under different ids.
SAME_AS_OUTER = [
    seg("k1", 0, 0, 10, 0),
    seg("k2", 10, 0, 10, 10),
    seg("k3", 10, 10, 0, 10),
    seg("k4", 0, 10, 0, 0),
]


# --- happy paths ----------------------------------------------------------------


def test_rectangle_with_hole_exact_response():
    resp = post(OUTER, HOLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outer"] == {
        "vertices": [[0, 0], [0, 10], [10, 10], [10, 0]],
        "edge_ids": ["o4", "o3", "o2", "o1"],
        "doubled_area": 200,
        "perimeter": 40,
    }
    assert body["hole"] == {
        "vertices": [[3, 3], [5, 3], [5, 5], [3, 5]],
        "edge_ids": ["h1", "h2", "h3", "h4"],
        "doubled_area": 8,
        "perimeter": 8,
    }
    assert body["net_doubled_area"] == 192
    assert body["total_cut_length"] == 48
    outer_vertices = verify_loop(body["outer"], OUTER, want_clockwise=True)
    verify_loop(body["hole"], HOLE, want_clockwise=False)
    verify_hole_strictly_inside(OUTER, HOLE, outer_vertices)


def test_concave_outer_with_hole_in_arm():
    resp = post(L_OUTER, L_HOLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outer"]["doubled_area"] == 96
    assert body["outer"]["perimeter"] == 32
    assert body["hole"]["doubled_area"] == 8
    assert body["net_doubled_area"] == 88
    assert body["total_cut_length"] == 40
    outer_vertices = verify_loop(body["outer"], L_OUTER, want_clockwise=True)
    verify_loop(body["hole"], L_HOLE, want_clockwise=False)
    verify_hole_strictly_inside(L_OUTER, L_HOLE, outer_vertices)


def test_split_edges_and_collinear_vertices():
    # Both loops carry 180-degree vertices: the outer bottom side is split
    # at (5,0) and the hole top side at (4,3).  The containment ray cast
    # from the hole's start vertex (3,1) runs exactly along y=1, and the
    # outer's split vertex must not confuse it.
    outer = [
        seg("b1", 0, 0, 5, 0),
        seg("b2", 5, 0, 10, 0),
        seg("b3", 10, 0, 10, 6),
        seg("b4", 10, 6, 0, 6),
        seg("b5", 0, 6, 0, 0),
    ]
    hole = [
        seg("u1", 3, 1, 7, 1),
        seg("u2", 7, 1, 7, 3),
        seg("u3", 7, 3, 4, 3),
        seg("u4", 4, 3, 3, 3),
        seg("u5", 3, 3, 3, 1),
    ]
    resp = post(outer, hole)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outer"]["doubled_area"] == 120
    assert body["hole"]["doubled_area"] == 16
    assert body["net_doubled_area"] == 104
    assert body["total_cut_length"] == 32 + 12
    outer_vertices = verify_loop(body["outer"], outer, want_clockwise=True)
    verify_loop(body["hole"], hole, want_clockwise=False)
    verify_hole_strictly_inside(outer, hole, outer_vertices)


def test_large_coordinates_exact_integer_areas():
    m = 10**6
    outer = [
        seg("g1", -m, -m, m, -m),
        seg("g2", m, -m, m, m),
        seg("g3", m, m, -m, m),
        seg("g4", -m, m, -m, -m),
    ]
    hole = [
        seg("c1", m - 3, m - 3, m - 1, m - 3),
        seg("c2", m - 1, m - 3, m - 1, m - 1),
        seg("c3", m - 1, m - 1, m - 3, m - 1),
        seg("c4", m - 3, m - 1, m - 3, m - 3),
    ]
    resp = post(outer, hole)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outer"]["doubled_area"] == 2 * (2 * m) ** 2  # 8_000_000_000_000
    assert body["hole"]["doubled_area"] == 8
    assert body["net_doubled_area"] == 2 * (2 * m) ** 2 - 8
    assert body["total_cut_length"] == 8 * m + 8
    outer_vertices = verify_loop(body["outer"], outer, want_clockwise=True)
    verify_loop(body["hole"], hole, want_clockwise=False)
    verify_hole_strictly_inside(outer, hole, outer_vertices)


def test_total_segment_count_at_limit():
    # 149 bottom + 149 top unit edges + 2 verticals = 300 outer edges;
    # 99 + 99 + 2 = 200 hole edges strictly inside: 500 in total.
    outer = [seg(f"ob{i}", i, 0, i + 1, 0) for i in range(149)]
    outer += [seg(f"ot{i}", i, 4, i + 1, 4) for i in range(149)]
    outer += [seg("ov0", 0, 0, 0, 4), seg("ov1", 149, 0, 149, 4)]
    hole = [seg(f"hb{i}", 10 + i, 1, 11 + i, 1) for i in range(99)]
    hole += [seg(f"ht{i}", 10 + i, 3, 11 + i, 3) for i in range(99)]
    hole += [seg("hv0", 10, 1, 10, 3), seg("hv1", 109, 1, 109, 3)]
    assert len(outer) + len(hole) == 500
    resp = post(outer, hole)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outer"]["doubled_area"] == 2 * 149 * 4
    assert body["hole"]["doubled_area"] == 2 * 99 * 2
    assert body["net_doubled_area"] == 2 * 149 * 4 - 2 * 99 * 2
    assert body["total_cut_length"] == 2 * (149 + 4) + 2 * (99 + 2)
    outer_vertices = verify_loop(body["outer"], outer, want_clockwise=True)
    verify_loop(body["hole"], hole, want_clockwise=False)
    verify_hole_strictly_inside(outer, hole, outer_vertices)


def test_responses_are_stable_under_shuffling():
    fixtures = [
        (OUTER, HOLE),
        (L_OUTER, L_HOLE),
        (OUTER, CROSS_HOLE),
        (OUTER, SHARED_EDGE_HOLE),
        (OUTER, POINT_TOUCH_HOLE),
        (OUTER, OUTSIDE_HOLE),
        (L_OUTER, L_STRADDLE_HOLE),
        (L_OUTER, L_NOTCH_HOLE),
    ]
    for outer, hole in fixtures:
        responses = [post(o, h).json() for o, h in shuffles(outer, hole)]
        first = responses[0]
        for other in responses[1:]:
            assert other == first


# --- hole placement rejections ---------------------------------------------------


def test_concave_outer_rejects_straddling_hole():
    # The hole's smallest vertex is inside the L (oracle-verified), so any
    # check that only tests one vertex would wrongly accept this request.
    assert oracle_point_inside((3, 3), L_POLYGON)
    assert not oracle_point_inside((6, 6), L_POLYGON)
    resp = post(L_OUTER, L_STRADDLE_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_INTERSECTS_OUTER"
    # crossing pairs are q2xL3 and q3xL4; the smallest involved id is L3
    assert expected_contact_witness(L_OUTER, L_STRADDLE_HOLE) == "L3"
    assert detail["witness"] == {"segment_id": "L3"}


def test_concave_outer_rejects_hole_in_notch():
    # Inside the bounding box, outside the solid: a bounding-box check or a
    # single-vertex test on the wrong vertex would misjudge this.
    assert_no_contact(L_OUTER, L_NOTCH_HOLE)
    assert not oracle_point_inside((5, 5), L_POLYGON)
    resp = post(L_OUTER, L_NOTCH_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_OUTSIDE_OUTER"
    assert detail["witness"] == {"segment_id": "n1"}


def test_hole_crossing_outer_edge_is_rejected():
    resp = post(OUTER, CROSS_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_INTERSECTS_OUTER"
    assert detail["witness"] == {"segment_id": expected_contact_witness(OUTER, CROSS_HOLE)}


def test_hole_sharing_edge_interval_is_rejected():
    resp = post(OUTER, SHARED_EDGE_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_INTERSECTS_OUTER"
    assert detail["witness"] == {
        "segment_id": expected_contact_witness(OUTER, SHARED_EDGE_HOLE)
    }


def test_hole_touching_at_single_point_is_rejected():
    # The only shared point of the two boundaries is the corner (10,10).
    resp = post(OUTER, POINT_TOUCH_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_INTERSECTS_OUTER"
    assert detail["witness"] == {
        "segment_id": expected_contact_witness(OUTER, POINT_TOUCH_HOLE)
    }


def test_hole_identical_to_outer_is_rejected():
    resp = post(OUTER, SAME_AS_OUTER)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_INTERSECTS_OUTER"
    # every edge of both loops is involved; the smallest id overall wins
    assert expected_contact_witness(OUTER, SAME_AS_OUTER) == "k1"
    assert detail["witness"] == {"segment_id": "k1"}


def test_hole_fully_outside_is_rejected():
    assert_no_contact(OUTER, OUTSIDE_HOLE)
    resp = post(OUTER, OUTSIDE_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_OUTSIDE_OUTER"
    assert detail["witness"] == {"segment_id": "w1"}


def test_hole_surrounding_outer_is_rejected():
    assert_no_contact(OUTER, SURROUNDING_HOLE)
    resp = post(OUTER, SURROUNDING_HOLE)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "HOLE_OUTSIDE_OUTER"
    assert detail["witness"] == {"segment_id": "r1"}


# --- per-group validation is reused ------------------------------------------------


def test_hole_group_input_validation():
    bad_hole = HOLE[:3] + [seg("h9", 3, 3, 4, 4)]
    resp = post(OUTER, bad_hole)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "NON_AXIS_ALIGNED"
    assert detail["witness"] == {"segment_id": "h9"}


def test_duplicate_id_across_groups():
    resp = post(OUTER, [seg("o1", 3, 3, 5, 3)] + HOLE[:3])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DUPLICATE_SEGMENT_ID"
    assert detail["witness"] == {"segment_id": "o1"}


def test_hole_group_bad_degree():
    resp = post(OUTER, HOLE + [seg("h9", 5, 5, 5, 7)])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "BAD_DEGREE"
    assert detail["witness"] == {"point": [5, 5]}


def test_hole_group_disconnected():
    two_squares = HOLE + [
        seg("z1", 6, 6, 7, 6),
        seg("z2", 7, 6, 7, 7),
        seg("z3", 7, 7, 6, 7),
        seg("z4", 6, 7, 6, 6),
    ]
    resp = post(OUTER, two_squares)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DISCONNECTED"
    assert detail["witness"] == {"segment_id": "z1"}


def test_hole_topology_is_checked_before_placement():
    # A self-intersecting hole loop far outside the outer contour:
    # SELF_INTERSECTION must win over HOLE_OUTSIDE_OUTER.
    crossing = [
        seg("e5", 24, 22, 24, 26),
        seg("e1", 22, 20, 22, 24),
        seg("e8", 26, 20, 22, 20),
        seg("e3", 20, 24, 20, 22),
        seg("e6", 24, 26, 26, 26),
        seg("e2", 22, 24, 20, 24),
        seg("e7", 26, 26, 26, 20),
        seg("e4", 20, 22, 24, 22),
    ]
    resp = post(OUTER, crossing)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "SELF_INTERSECTION"


def test_outer_topology_error_beats_hole_topology_error():
    # Both groups broken: the outer group's failure is reported first.
    bad_outer = OUTER + [seg("o9", 5, 10, 5, 12)]
    bad_hole = HOLE + [seg("h9", 5, 5, 5, 7)]
    resp = post(bad_outer, bad_hole)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "BAD_DEGREE"
    assert detail["witness"] == {"point": [5, 10]}


def test_input_validation_beats_topology():
    # A malformed hole segment is an input error and outranks the outer
    # group's topology failure.
    bad_outer = OUTER + [seg("o9", 5, 10, 5, 12)]
    resp = post(bad_outer, HOLE[:3] + [seg("h9", 3, 3, 4, 4)])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "NON_AXIS_ALIGNED"


# --- request shape and counts -------------------------------------------------------


def test_malformed_payload():
    resp = client.post("/reconstruct-with-hole", json={"segments": OUTER})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

    resp = client.post("/reconstruct-with-hole", json={"outer": OUTER})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

    resp = client.post("/reconstruct-with-hole", json={"outer": "x", "hole": HOLE})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"


def test_segment_count_bounds():
    resp = post(OUTER[:3], HOLE)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"
    assert resp.json()["detail"]["witness"] == {"count": 3, "group": "outer"}

    resp = post(OUTER, HOLE[:3])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"
    assert resp.json()["detail"]["witness"] == {"count": 3, "group": "hole"}

    # Each group within 4..500 but the total above the shared limit.
    outer = [seg(f"a{i}", i, 0, i + 1, 0) for i in range(300)]
    hole = [seg(f"b{i}", i, 2, i + 1, 2) for i in range(300)]
    resp = post(outer, hole)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"
    assert resp.json()["detail"]["witness"] == {"count": 600}


# --- the original single-loop endpoint is untouched -----------------------------------


def test_single_loop_endpoint_is_unchanged():
    resp = client.post("/reconstruct", json={"segments": OUTER})
    assert resp.status_code == 200
    assert resp.json() == {
        "vertices": [[0, 0], [0, 10], [10, 10], [10, 0]],
        "edge_ids": ["o4", "o3", "o2", "o1"],
        "doubled_area": 200,
        "perimeter": 40,
    }

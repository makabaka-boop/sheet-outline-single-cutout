"""End-to-end tests: shuffled integer-geometry fixtures against the HTTP API."""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def seg(seg_id, x1, y1, x2, y2):
    return {"id": seg_id, "a": [x1, y1], "b": [x2, y2]}


def post(segments):
    return client.post("/reconstruct", json={"segments": segments})


# --- fixtures ---------------------------------------------------------------

# 2x2 square; ids deliberately shuffled so traversal order must be derived,
# not copied from the input order.
SQUARE = [
    seg("e4", 0, 0, 2, 0),
    seg("e1", 2, 0, 2, 2),
    seg("e3", 2, 2, 0, 2),
    seg("e2", 0, 2, 0, 0),
]

# Concave L-shape, reflex vertex at (2,2).
CONCAVE = [
    seg("k2", 0, 0, 4, 0),
    seg("k5", 4, 0, 4, 2),
    seg("k1", 4, 2, 2, 2),
    seg("k6", 2, 2, 2, 4),
    seg("k3", 2, 4, 0, 4),
    seg("k4", 0, 4, 0, 0),
]

# Rectangle whose bottom side is split into two collinear edges: the vertex
# (2,0) is a 180-degree corner and must survive reconstruction.
SPLIT_EDGE = [
    seg("a1", 0, 0, 2, 0),
    seg("a2", 2, 0, 4, 0),
    seg("a3", 4, 0, 4, 2),
    seg("a4", 4, 2, 0, 2),
    seg("a5", 0, 2, 0, 0),
]

# Single 8-edge cycle, connected, all degrees 2 -- but edges e1 and e4
# cross at (2,2).
SELF_CROSSING = [
    seg("e5", 4, 2, 4, 6),
    seg("e1", 2, 0, 2, 4),
    seg("e8", 6, 0, 2, 0),
    seg("e3", 0, 4, 0, 2),
    seg("e6", 4, 6, 6, 6),
    seg("e2", 2, 4, 0, 4),
    seg("e7", 6, 6, 6, 0),
    seg("e4", 0, 2, 4, 2),
]

# Two squares touching only at the corner (2,2): that endpoint has degree 4.
CORNER_TOUCH = [
    seg("c1", 0, 0, 2, 0),
    seg("c2", 2, 0, 2, 2),
    seg("c3", 2, 2, 0, 2),
    seg("c4", 0, 2, 0, 0),
    seg("c5", 2, 2, 4, 2),
    seg("c6", 4, 2, 4, 4),
    seg("c7", 4, 4, 2, 4),
    seg("c8", 2, 4, 2, 2),
]

# Two disjoint unit squares: every degree is 2, but there are two components.
MULTI_LOOP = [
    seg("m1", 0, 0, 1, 0),
    seg("m2", 1, 0, 1, 1),
    seg("m3", 1, 1, 0, 1),
    seg("m4", 0, 1, 0, 0),
    seg("m5", 5, 5, 6, 5),
    seg("m6", 6, 5, 6, 6),
    seg("m7", 6, 6, 5, 6),
    seg("m8", 5, 6, 5, 5),
]

# A square plus a dangling edge ending on the middle of its top side:
# (2,2) has degree 3, (2,4) has degree 1.
T_CONTACT = [
    seg("t1", 0, 0, 4, 0),
    seg("t2", 4, 0, 4, 2),
    seg("t3", 4, 2, 0, 2),
    seg("t4", 0, 2, 0, 0),
    seg("t5", 2, 2, 2, 4),
]


def shuffles(segments, rounds=5):
    """Deterministic set of permutations of the same batch."""
    orders = [list(segments), list(reversed(segments))]
    rng = random.Random(20260923)
    for _ in range(rounds):
        perm = list(segments)
        rng.shuffle(perm)
        orders.append(perm)
    return orders


# --- independent recomputation ----------------------------------------------


def verify_contour(body, segments):
    """Re-derive every guarantee of a 200 response from the raw input."""
    vertices = [tuple(v) for v in body["vertices"]]
    edge_ids = body["edge_ids"]
    by_id = {s["id"]: s for s in segments}
    n = len(vertices)

    assert n == len(segments) == len(edge_ids)
    assert set(edge_ids) == set(by_id)  # every edge used exactly once
    assert vertices[0] == min(vertices)  # lexicographically smallest start

    # edge i must connect vertices[i] to vertices[i+1]
    for i in range(n):
        s = by_id[edge_ids[i]]
        ends = {tuple(s["a"]), tuple(s["b"])}
        assert ends == {vertices[i], vertices[(i + 1) % n]}

    # integer shoelace, recomputed from scratch
    signed = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    assert signed < 0  # clockwise in standard Cartesian orientation
    assert body["doubled_area"] == -signed

    perimeter = sum(
        abs(s["a"][0] - s["b"][0]) + abs(s["a"][1] - s["b"][1]) for s in segments
    )
    assert body["perimeter"] == perimeter


# --- happy paths -------------------------------------------------------------


def test_square_exact_response():
    resp = post(SQUARE)
    assert resp.status_code == 200
    body = resp.json()
    # start (0,0); smallest incident id is "e2", and that direction is
    # already clockwise, so no flip happens.
    assert body["vertices"] == [[0, 0], [0, 2], [2, 2], [2, 0]]
    assert body["edge_ids"] == ["e2", "e3", "e1", "e4"]
    assert body["doubled_area"] == 8
    assert body["perimeter"] == 8
    verify_contour(body, SQUARE)


def test_concave_polygon_exact_response():
    resp = post(CONCAVE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["vertices"] == [[0, 0], [0, 4], [2, 4], [2, 2], [4, 2], [4, 0]]
    assert body["edge_ids"] == ["k4", "k3", "k6", "k1", "k5", "k2"]
    assert body["doubled_area"] == 24
    assert body["perimeter"] == 16
    verify_contour(body, CONCAVE)


def test_collinear_split_edge_is_kept():
    resp = post(SPLIT_EDGE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["vertices"] == [[0, 0], [0, 2], [4, 2], [4, 0], [2, 0]]
    assert body["edge_ids"] == ["a5", "a4", "a3", "a2", "a1"]
    assert body["doubled_area"] == 16
    assert body["perimeter"] == 12
    verify_contour(body, SPLIT_EDGE)


def test_negative_coordinates():
    segments = [
        seg("w1", -3, -1, -1, -1),
        seg("w2", -1, -1, -1, 2),
        seg("w3", -1, 2, -3, 2),
        seg("w4", -3, 2, -3, -1),
    ]
    resp = post(segments)
    assert resp.status_code == 200
    body = resp.json()
    assert body["vertices"][0] == [-3, -1]
    assert body["doubled_area"] == 12
    assert body["perimeter"] == 10
    verify_contour(body, segments)


def test_max_segment_count_boundary():
    # 249 bottom + 249 top unit edges + 2 verticals = 500 edges.
    segments = [seg(f"b{i}", i, 0, i + 1, 0) for i in range(249)]
    segments += [seg(f"t{i}", i, 2, i + 1, 2) for i in range(249)]
    segments += [seg("v0", 0, 0, 0, 2), seg("v1", 249, 0, 249, 2)]
    assert len(segments) == 500
    resp = post(segments)
    assert resp.status_code == 200
    body = resp.json()
    assert body["doubled_area"] == 2 * 249 * 2
    assert body["perimeter"] == 2 * (249 + 2)
    verify_contour(body, segments)


def test_output_is_stable_under_shuffling():
    for fixture in (SQUARE, CONCAVE, SPLIT_EDGE, SELF_CROSSING, MULTI_LOOP):
        responses = [post(order).json() for order in shuffles(fixture)]
        first = responses[0]
        for other in responses[1:]:
            assert other == first


# --- topology rejections (deterministic witnesses) ----------------------------


def test_bad_degree_from_corner_touch():
    resp = post(CORNER_TOUCH)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "BAD_DEGREE"
    assert detail["witness"] == {"point": [2, 2]}


def test_bad_degree_from_t_contact():
    resp = post(T_CONTACT)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "BAD_DEGREE"
    # degree-3 (2,2) is lexicographically smaller than degree-1 (2,4)
    assert detail["witness"] == {"point": [2, 2]}


def test_disconnected_multi_loop():
    resp = post(MULTI_LOOP)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DISCONNECTED"
    # smallest edge id outside the main component
    assert detail["witness"] == {"segment_id": "m5"}


def test_self_intersection_single_cycle():
    resp = post(SELF_CROSSING)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "SELF_INTERSECTION"
    # crossing pair is {e1, e4}; smallest id wins
    assert detail["witness"] == {"segment_id": "e1"}


def test_failure_order_degree_before_disconnected():
    # Two components AND a degree-3 vertex: degree must be reported first.
    segments = T_CONTACT + [
        seg("u1", 10, 10, 11, 10),
        seg("u2", 11, 10, 11, 11),
        seg("u3", 11, 11, 10, 11),
        seg("u4", 10, 11, 10, 10),
    ]
    resp = post(segments)
    assert resp.json()["detail"]["code"] == "BAD_DEGREE"


def test_failure_order_disconnected_before_self_intersection():
    # One clean square plus a separate self-crossing cycle (shifted so the
    # two share no endpoints: degrees stay 2 everywhere).
    shifted = [
        seg(f"z{s['id']}", s["a"][0] + 10, s["a"][1], s["b"][0] + 10, s["b"][1])
        for s in SELF_CROSSING
    ]
    resp = post(SQUARE + shifted)
    assert resp.json()["detail"]["code"] == "DISCONNECTED"
    assert resp.json()["detail"]["witness"] == {"segment_id": "ze1"}


# --- input validation (422) ---------------------------------------------------


def test_zero_length_segment():
    resp = post(SQUARE[:3] + [seg("z5", 1, 1, 1, 1)])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "ZERO_LENGTH_SEGMENT"
    assert detail["witness"] == {"segment_id": "z5"}


def test_duplicate_edge():
    segments = SQUARE + [seg("d5", 2, 0, 0, 0)]  # same span as e4, reversed
    resp = post(segments)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DUPLICATE_EDGE"
    assert detail["witness"] == {"segment_id": "d5", "other_segment_id": "e4"}


def test_partial_overlap():
    segments = SQUARE + [seg("p5", 1, 0, 3, 0)]  # overlaps e4 on [1, 2]
    resp = post(segments)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "PARTIAL_OVERLAP"
    assert detail["witness"] == {"segment_id": "e4", "other_segment_id": "p5"}


def test_non_axis_aligned():
    resp = post(SQUARE[:3] + [seg("n5", 0, 0, 1, 1)])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "NON_AXIS_ALIGNED"


def test_coordinate_out_of_range():
    resp = post(SQUARE[:3] + [seg("o5", 0, 0, 10**6 + 1, 0)])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "COORDINATE_OUT_OF_RANGE"


def test_coordinate_at_limit_is_accepted():
    m = 10**6
    segments = [
        seg("g1", -m, -m, m, -m),
        seg("g2", m, -m, m, m),
        seg("g3", m, m, -m, m),
        seg("g4", -m, m, -m, -m),
    ]
    resp = post(segments)
    assert resp.status_code == 200
    assert resp.json()["doubled_area"] == 2 * (2 * m) ** 2


def test_non_integer_coordinate():
    resp = post(SQUARE[:3] + [seg("q5", 0, 0, 1, 0.5)])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_POINT"


def test_duplicate_segment_id():
    segments = SQUARE + [seg("e1", 5, 5, 6, 5)]
    resp = post(segments)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "DUPLICATE_SEGMENT_ID"
    assert detail["witness"] == {"segment_id": "e1"}


def test_segment_count_bounds():
    resp = post(SQUARE[:3])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"

    too_many = [seg(f"s{i}", i, 0, i + 1, 0) for i in range(501)]
    resp = post(too_many)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_SEGMENT_COUNT"


def test_malformed_payload():
    resp = client.post("/reconstruct", json={"edges": []})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"

    resp = client.post("/reconstruct", json=[1, 2, 3])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PAYLOAD"


def test_validation_witnesses_are_shuffle_stable():
    fixture = SQUARE + [seg("p5", 1, 0, 3, 0)]
    details = [post(order).json()["detail"] for order in shuffles(fixture)]
    assert all(d == details[0] for d in details)

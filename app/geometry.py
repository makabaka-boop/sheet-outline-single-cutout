"""Pure integer-geometry reconstruction of a closed orthogonal outer contour.

No floating point is used anywhere: all predicates and all outputs
(doubled area, perimeter) are exact integers.

Pipeline (deterministic, order-sensitive):

1. Request validation  -> 422 with a stable ``code`` + witness.
2. Endpoint adjacency reconstruction:
   every endpoint must have degree 2          -> BAD_DEGREE
   all edges must form a single component     -> DISCONNECTED
   non-adjacent edges must not intersect/touch-> SELF_INTERSECTION
3. Traversal from the lexicographically smallest vertex, normalised to
   clockwise (standard Cartesian orientation, shoelace sign), plus the
   integer shoelace doubled area and the perimeter.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SEGMENTS = 4
MAX_SEGMENTS = 500
COORD_LIMIT = 10**6


class ReconstructionError(Exception):
    """Deterministic rejection carrying a machine-checkable witness."""

    def __init__(self, code: str, message: str, witness: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.witness = witness or {}


@dataclass(frozen=True)
class Segment:
    seg_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.x1, self.y1), (self.x2, self.y2)


# ---------------------------------------------------------------------------
# Request validation (everything here is rejected with HTTP 422)
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    # bool is a subclass of int; coordinates must be genuine integers.
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_point(value: object, seg_id: object) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(_is_int(v) for v in value)
    ):
        raise ReconstructionError(
            "INVALID_POINT",
            f"segment {seg_id!r}: endpoint must be a pair of integers",
            {"segment_id": seg_id},
        )
    x, y = value
    if abs(x) > COORD_LIMIT or abs(y) > COORD_LIMIT:
        raise ReconstructionError(
            "COORDINATE_OUT_OF_RANGE",
            f"segment {seg_id!r}: coordinates must satisfy |v| <= {COORD_LIMIT}",
            {"segment_id": seg_id},
        )
    return (x, y)


def parse_segments(payload: object) -> list[Segment]:
    """Validate the raw JSON payload and return normalised segments.

    Every violation raises :class:`ReconstructionError` whose code is served
    with HTTP 422.  Checks run in a fixed order and witnesses are chosen by
    ascending id / lexicographic order, so a rejected batch always fails with
    the exact same evidence regardless of how the edges were shuffled.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ReconstructionError(
            "INVALID_PAYLOAD",
            'request body must be an object with a "segments" array',
        )
    raw = payload["segments"]
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        raise ReconstructionError(
            "INVALID_SEGMENT_COUNT",
            f"expected {MIN_SEGMENTS}..{MAX_SEGMENTS} segments, got {len(raw)}",
            {"count": len(raw)},
        )

    seen_ids: set[str] = set()
    segments: list[Segment] = []
    for item in raw:
        seg_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(seg_id, str) or not seg_id:
            raise ReconstructionError(
                "INVALID_SEGMENT_ID",
                "every segment needs a non-empty string id",
                {"segment_id": seg_id},
            )
        if seg_id in seen_ids:
            raise ReconstructionError(
                "DUPLICATE_SEGMENT_ID",
                f"duplicate segment id {seg_id!r}",
                {"segment_id": seg_id},
            )
        seen_ids.add(seg_id)
        a = _parse_point(item.get("a"), seg_id)
        b = _parse_point(item.get("b"), seg_id)
        if a == b:
            raise ReconstructionError(
                "ZERO_LENGTH_SEGMENT",
                f"segment {seg_id!r} has zero length",
                {"segment_id": seg_id},
            )
        if a[0] != b[0] and a[1] != b[1]:
            raise ReconstructionError(
                "NON_AXIS_ALIGNED",
                f"segment {seg_id!r} is neither horizontal nor vertical",
                {"segment_id": seg_id},
            )
        segments.append(Segment(seg_id, a[0], a[1], b[0], b[1]))

    _reject_collinear_overlaps(segments)
    return segments


def _canonical(seg: Segment) -> tuple[str, int, int, int]:
    """(axis, fixed coordinate, interval lo, interval hi) with lo < hi."""
    if seg.y1 == seg.y2:
        lo, hi = sorted((seg.x1, seg.x2))
        return ("h", seg.y1, lo, hi)
    lo, hi = sorted((seg.y1, seg.y2))
    return ("v", seg.x1, lo, hi)


def _reject_collinear_overlaps(segments: list[Segment]) -> None:
    """Duplicate edges and partial overlaps are input errors (422).

    A positive-length shared interval between two collinear segments is
    ambiguous for reconstruction, so the whole batch is refused.  Identical
    intervals are reported as DUPLICATE_EDGE, anything else as
    PARTIAL_OVERLAP.  Each pass scans pairs in ascending id order, which
    keeps the witness stable under input shuffling.
    """
    ordered = sorted(segments, key=lambda s: s.seg_id)
    canon = [(s.seg_id, _canonical(s)) for s in ordered]

    def scan(want_duplicate: bool) -> None:
        for i in range(len(canon)):
            id_i, (axis_i, fixed_i, lo_i, hi_i) = canon[i]
            for j in range(i + 1, len(canon)):
                id_j, (axis_j, fixed_j, lo_j, hi_j) = canon[j]
                if axis_i != axis_j or fixed_i != fixed_j:
                    continue
                overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
                if overlap <= 0:
                    continue
                is_duplicate = lo_i == lo_j and hi_i == hi_j
                if is_duplicate != want_duplicate:
                    continue
                first, second = sorted((id_i, id_j))
                code = "DUPLICATE_EDGE" if is_duplicate else "PARTIAL_OVERLAP"
                raise ReconstructionError(
                    code,
                    f"segments {first!r} and {second!r} share a collinear interval",
                    {"segment_id": first, "other_segment_id": second},
                )

    scan(want_duplicate=True)
    scan(want_duplicate=False)


# ---------------------------------------------------------------------------
# Topology reconstruction
# ---------------------------------------------------------------------------


def _check_degrees(incidence: dict[tuple[int, int], list[Segment]]) -> None:
    bad = sorted(p for p, segs in incidence.items() if len(segs) != 2)
    if bad:
        x, y = bad[0]
        raise ReconstructionError(
            "BAD_DEGREE",
            f"endpoint ({x}, {y}) has degree {len(incidence[(x, y)])}, expected 2",
            {"point": [x, y]},
        )


def _check_connected(segments: list[Segment]) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(p: tuple[int, int]) -> tuple[int, int]:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for seg in segments:
        a, b = seg.endpoints
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    # The main component is the one containing the lexicographically
    # smallest endpoint; the witness is the smallest segment id outside it.
    main_root = find(min(parent))
    outsiders = sorted(
        seg.seg_id for seg in segments if find(seg.endpoints[0]) != main_root
    )
    if outsiders:
        raise ReconstructionError(
            "DISCONNECTED",
            f"segment {outsiders[0]!r} is not connected to the main component",
            {"segment_id": outsiders[0]},
        )


def _segments_cross(s: Segment, t: Segment) -> bool:
    """Proper or touching intersection of an H and a V segment."""
    if s.y1 == s.y2:  # s horizontal, t vertical
        h, v = s, t
    else:
        h, v = t, s
    hx_lo, hx_hi = sorted((h.x1, h.x2))
    vy_lo, vy_hi = sorted((v.y1, v.y2))
    return hx_lo <= v.x1 <= hx_hi and vy_lo <= h.y1 <= vy_hi


def _check_self_intersection(segments: list[Segment]) -> None:
    """Non-adjacent edges must neither cross nor touch.

    Collinear pairs were already normalised by validation: they are either
    disjoint or share exactly one endpoint (adjacent), so only H/V pairs can
    violate anything here.
    """
    bad_ids: set[str] = set()
    for i in range(len(segments)):
        s = segments[i]
        s_ends = set(s.endpoints)
        for j in range(i + 1, len(segments)):
            t = segments[j]
            if (s.y1 == s.y2) == (t.y1 == t.y2):
                continue  # parallel pairs are already settled
            if s_ends & set(t.endpoints):
                continue  # adjacent edges legitimately share an endpoint
            if _segments_cross(s, t):
                bad_ids.add(s.seg_id)
                bad_ids.add(t.seg_id)
    if bad_ids:
        culprit = min(bad_ids)
        raise ReconstructionError(
            "SELF_INTERSECTION",
            f"non-adjacent edges intersect or touch (smallest id: {culprit!r})",
            {"segment_id": culprit},
        )


# ---------------------------------------------------------------------------
# Traversal and measurements
# ---------------------------------------------------------------------------


def _trace(
    segments: list[Segment], incidence: dict[tuple[int, int], list[Segment]]
) -> tuple[list[tuple[int, int]], list[str]]:
    start = min(incidence)
    # Deterministic tie-break: leave the start vertex along the smaller id.
    first = min(incidence[start], key=lambda s: s.seg_id)

    vertices = [start]
    edge_ids: list[str] = []
    prev, seg = start, first
    while True:
        edge_ids.append(seg.seg_id)
        a, b = seg.endpoints
        current = b if a == prev else a
        if current == start:
            break
        vertices.append(current)
        followers = [s for s in incidence[current] if s.seg_id != seg.seg_id]
        if len(followers) != 1:  # pragma: no cover - guarded by earlier checks
            raise ReconstructionError(
                "BAD_DEGREE", "contour traversal hit a branching vertex"
            )
        prev, seg = current, followers[0]
    if len(edge_ids) != len(segments):  # pragma: no cover - guarded earlier
        raise ReconstructionError("DISCONNECTED", "traversal did not cover all edges")
    return vertices, edge_ids


def _signed_doubled_area(vertices: list[tuple[int, int]]) -> int:
    total = 0
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
        total += x1 * y2 - x2 * y1
    return total


def reconstruct(segments: list[Segment]) -> dict:
    """Rebuild the closed contour from validated segments.

    Returns ``vertices`` (clockwise, starting at the lexicographically
    smallest vertex), the ``edge_ids`` travelled alongside, the integer
    ``doubled_area`` from the shoelace formula and the ``perimeter``.
    """
    incidence: dict[tuple[int, int], list[Segment]] = {}
    for seg in segments:
        for p in seg.endpoints:
            incidence.setdefault(p, []).append(seg)

    _check_degrees(incidence)
    _check_connected(segments)
    _check_self_intersection(segments)

    vertices, edge_ids = _trace(segments, incidence)
    signed_area = _signed_doubled_area(vertices)
    doubled_area = abs(signed_area)
    if signed_area > 0:
        # Counter-clockwise in standard Cartesian orientation: flip to
        # clockwise while keeping the start vertex first.
        vertices = [vertices[0]] + vertices[:0:-1]
        edge_ids.reverse()

    perimeter = sum(
        abs(seg.x2 - seg.x1) + abs(seg.y2 - seg.y1) for seg in segments
    )
    return {
        "vertices": [[x, y] for x, y in vertices],
        "edge_ids": edge_ids,
        "doubled_area": doubled_area,
        "perimeter": perimeter,
    }

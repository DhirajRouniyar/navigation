
import numpy as np

def extract_trees(scan, params):
    max_range = 75
    min_range = 1
    min_angle = 5 * np.pi / 306
    range_jump = 1.5
    angle_jump = 10 * np.pi / 360
    max_pair_distance = 3
    max_diameter = 1
    min_angle_between_segments = 2 * np.pi / 360

    angles = np.linspace(0, np.pi, len(scan))
    valid = scan < max_range
    scan = scan[valid]
    angles = angles[valid]

    if len(scan) < 2:
        return []

    # Find segmentation points where range or angle jumps
    segment_indices = np.flatnonzero(
        (np.abs(np.diff(scan)) > range_jump) | 
        (np.diff(angles) > angle_jump)
    )
    segment_starts = np.insert(segment_indices + 1, 0, 0)
    segment_ends = np.append(segment_indices, len(scan) - 1)

    if len(segment_starts) < 2:
        return []

    # Get coordinates of segment start and end points
    ranges_start = scan[segment_starts]
    ranges_end = scan[segment_ends]
    angles_start = angles[segment_starts]
    angles_end = angles[segment_ends]
    x_start = ranges_start * np.cos(angles_start)
    y_start = ranges_start * np.sin(angles_start)
    x_end = ranges_end * np.cos(angles_end)
    y_end = ranges_end * np.sin(angles_end)

    keep = np.ones(len(segment_starts), dtype=bool)

    # Filter segments too close together
    for offset in range(1, 4):
        if len(segment_starts) > offset:
            dx = x_start[offset:] - x_end[:-offset]
            dy = y_start[offset:] - y_end[:-offset]
            dist_sq = dx**2 + dy**2
            close = dist_sq < max_pair_distance**2
            keep[offset:][close] = False
            keep[:-offset][close] = False

    # Filter segments that are close in angle but inconsistent in depth
    angle_diff = angles_start[1:] - angles_end[:-1]
    near = angle_diff < min_angle_between_segments
    if np.any(near):
        deeper = ranges_start[1:][near] > ranges_end[:-1][near]
        idx = np.flatnonzero(near) + deeper
        keep[idx] = False

    # Apply keep mask
    segment_starts = segment_starts[keep]
    segment_ends = segment_ends[keep]
    ranges_start = ranges_start[keep]
    ranges_end = ranges_end[keep]
    angles_start = angles_start[keep]
    angles_end = angles_end[keep]

    if len(segment_starts) == 0:
        return []

    # Compute segment midpoint distances
    dx = ranges_start * np.cos(angles_start) - ranges_end * np.cos(angles_end)
    dy = ranges_start * np.sin(angles_start) - ranges_end * np.sin(angles_end)
    segment_diameters = np.sqrt(dx**2 + dy**2)

    # Filter by diameter
    small_enough = segment_diameters < max_diameter
    if not np.any(small_enough):
        return []

    # Apply diameter filter
    ranges_start = ranges_start[small_enough]
    ranges_end = ranges_end[small_enough]
    angles_start = angles_start[small_enough]
    angles_end = angles_end[small_enough]
    segment_diameters = segment_diameters[small_enough]
    segment_starts = segment_starts[small_enough]
    segment_ends = segment_ends[small_enough]

    # Filter out noisy edge detections
    within_limits = (
        (ranges_start > min_range) &
        (angles_start > min_angle) &
        (angles_end < (np.pi - min_angle))
    )
    if not np.any(within_limits):
        return []

    ranges_start = ranges_start[within_limits]
    ranges_end = ranges_end[within_limits]
    angles_start = angles_start[within_limits]
    angles_end = angles_end[within_limits]
    segment_diameters = segment_diameters[within_limits]
    segment_starts = segment_starts[within_limits]
    segment_ends = segment_ends[within_limits]

    # Consistency check
    arc_length = (angles_end - angles_start) * (ranges_start + ranges_end) / 2
    consistent = np.abs(ranges_start - ranges_end) < arc_length / 3
    if not np.any(consistent):
        return []

    # Final estimation
    mid_indices = ((segment_starts + segment_ends) / 2).astype(int)
    mid_indices = mid_indices[consistent]
    final_ranges = (scan[mid_indices] + arc_length[consistent] / 2)
    final_angles = (angles_start[consistent] + angles_end[consistent]) / 2 - np.pi / 2
    final_diameters = arc_length[consistent]

    return list(zip(final_ranges, final_angles, final_diameters))

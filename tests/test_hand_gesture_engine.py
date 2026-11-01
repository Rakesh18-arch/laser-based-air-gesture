"""Hardware-independent regression tests for AIRWRITE's hand gesture geometry."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1] / "desktop"
sys.path.insert(0, str(DESKTOP))

from hand_gesture_engine import (  # noqa: E402
    GestureConsensus,
    OneEuroPointFilter,
    classify_hand_pose,
    select_primary_hand,
    map_hand_pointer_to_canvas,
)


def base_hand():
    """A normalized 21-landmark right hand, palm facing the camera."""
    p = [[0.50, 0.78, 0.0] for _ in range(21)]
    p[0] = [0.50, 0.82, 0.0]
    p[1], p[2], p[3], p[4] = [0.43, 0.67, 0.0], [0.39, 0.62, 0.0], [0.35, 0.57, 0.0], [0.31, 0.54, 0.0]
    for base, x in ((5, 0.40), (9, 0.50), (13, 0.59), (17, 0.68)):
        p[base] = [x, 0.58, 0.0]
    for mcp, pip, tip, x in (
        (5, 6, 8, 0.40), (9, 10, 12, 0.50),
        (13, 14, 16, 0.59), (17, 18, 20, 0.68),
    ):
        p[pip] = [x, 0.43, 0.0]
        p[tip] = [x, 0.26, 0.0]
    return p


def fold(p, names):
    ids = {
        "middle": (9, 10, 12),
        "ring": (13, 14, 16),
        "pinky": (17, 18, 20),
        "index": (5, 6, 8),
    }
    for name in names:
        mcp, pip, tip = ids[name]
        x, y, z = p[mcp]
        p[pip] = [x + 0.035, y + 0.025, z]
        p[tip] = [x + 0.065, y + 0.105, z]
    return p


def mediapipe_landmarks(points):
    return [types.SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


class HandGestureEngineTests(unittest.TestCase):
    def pose(self, points, pinch=False, zoom=False):
        if pinch:
            points[4] = points[8][:]
        if zoom:
            points[4] = points[12][:]
            # Keep the index away from the thumb so thumb-middle is unambiguous.
            points[8] = [0.22, 0.18, 0.0]
        return classify_hand_pose(mediapipe_landmarks(points), 640, 480)

    def test_index_tip_is_the_only_pointer_even_with_other_fingers_curled(self):
        pose = self.pose(fold(base_hand(), ["middle", "ring", "pinky"]))
        self.assertTrue(pose.index_extended)
        self.assertTrue(pose.pointer_active)
        self.assertEqual(pose.gesture, "POINT · MOVE")
        self.assertEqual(pose.index_point, (0.40 * 640, 0.26 * 480))

    def test_thumb_index_pinch_draws_without_open_palm(self):
        points = fold(base_hand(), ["middle", "ring", "pinky"])
        pose = self.pose(points, pinch=True)
        self.assertTrue(pose.pinch_active)
        self.assertTrue(pose.pointer_active)
        self.assertEqual(pose.gesture, "PINCH · DRAW / SELECT")

    def test_v_sign_pan(self):
        pose = self.pose(fold(base_hand(), ["ring", "pinky"]))
        self.assertTrue(pose.pan_active)
        self.assertEqual(pose.gesture, "PAN · V SIGN")

    def test_three_raised_fingers_open_tools_menu(self):
        pose = self.pose(fold(base_hand(), ["pinky"]))
        self.assertTrue(pose.menu_active)
        self.assertEqual(pose.gesture, "THREE FINGERS · TOOLS")

    def test_thumb_middle_zoom_keeps_pointer_mapping_live(self):
        pose = self.pose(base_hand(), zoom=True)
        self.assertTrue(pose.zoom_active)
        self.assertTrue(pose.pointer_active)
        self.assertEqual(pose.gesture, "ZOOM · THUMB-MIDDLE")

    def test_finger_extension_is_invariant_to_hand_rotation(self):
        points = fold(base_hand(), ["middle", "ring", "pinky"])
        # Rotate the complete hand 55 degrees around the wrist. The pointer
        # should remain index-tip based rather than relying on vertical image y.
        ox, oy, _ = points[0]
        import math
        angle = math.radians(55)
        rotated = []
        for x, y, z in points:
            dx, dy = x - ox, y - oy
            rotated.append([
                ox + dx * math.cos(angle) - dy * math.sin(angle),
                oy + dx * math.sin(angle) + dy * math.cos(angle), z
            ])
        pose = self.pose(rotated)
        self.assertTrue(pose.index_extended)

    def test_primary_hand_selection_stays_with_nearest_track(self):
        a = mediapipe_landmarks(base_hand())
        shifted = [[min(0.99, x + 0.22), y, z] for x, y, z in base_hand()]
        b = mediapipe_landmarks(shifted)
        hand_a = types.SimpleNamespace(landmark=a)
        hand_b = types.SimpleNamespace(landmark=b)
        selected, center = select_primary_hand([hand_a, hand_b], 640, 480)
        self.assertIsNotNone(selected)
        # When a track is already established, do not jump to a new hand merely
        # because it enters the frame; stick with the hand nearest that track.
        target_center = ((0.72 + 0.72) * 0.5 * 640, (0.82 + 0.58) * 0.5 * 480)
        selected, _ = select_primary_hand([hand_a, hand_b], 640, 480, previous_center=target_center)
        self.assertIs(selected, hand_b)

    def test_pan_and_menu_need_three_of_five_frames(self):
        gate = GestureConsensus(window=5, min_votes=3)
        self.assertEqual(gate.update(True, False), (False, False))
        self.assertEqual(gate.update(False, False), (False, False))
        self.assertEqual(gate.update(True, False), (False, False))
        self.assertEqual(gate.update(True, False), (True, False))
        self.assertEqual(gate.update(False, True), (True, False))
        self.assertEqual(gate.update(False, True), (False, False))
        self.assertEqual(gate.update(False, True), (False, True))

    def test_one_euro_filter_smooths_small_coordinate_jitter(self):
        f = OneEuroPointFilter()
        out = [f.update(pt, i * 0.04) for i, pt in enumerate(
            [(100, 100), (102, 99), (99, 101), (101, 100), (100, 99)]
        )]
        self.assertTrue(all(abs(x - 100) < 3 and abs(y - 100) < 3 for x, y in out))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestNoCalibrationHandMapping:
    def test_center_maps_to_canvas_center(self):
        point = map_hand_pointer_to_canvas((320, 240), (640, 480), (100, 50, 800, 600))
        assert point == (500.0, 350.0)

    def test_mirrored_edges_map_to_opposite_canvas_edges(self):
        left = map_hand_pointer_to_canvas((0, 0), (640, 480), (10, 20, 900, 500))
        right = map_hand_pointer_to_canvas((640, 480), (640, 480), (10, 20, 900, 500))
        assert left == (910.0, 20.0)
        assert right == (10.0, 520.0)

    def test_point_is_clamped_to_canvas(self):
        point = map_hand_pointer_to_canvas((900, -20), (640, 480), (0, 0, 400, 300))
        assert point == (0.0, 0.0)

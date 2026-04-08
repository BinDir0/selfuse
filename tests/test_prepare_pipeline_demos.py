import unittest

from tools.ops.prepare_pipeline_demos import parse_render_modes, peak_score, stable_inverse_score


class PreparePipelineDemosTests(unittest.TestCase):
    def test_parse_render_modes_deduplicates_and_normalizes(self):
        self.assertEqual(parse_render_modes(" keypoint,mano,keypoint "), ["keypoint", "mano"])

    def test_peak_score_peaks_inside_interval(self):
        self.assertEqual(peak_score(0.5, low=0.0, peak=0.5, high=1.0), 1.0)
        self.assertEqual(peak_score(-1.0, low=0.0, peak=0.5, high=1.0), 0.0)
        self.assertEqual(peak_score(2.0, low=0.0, peak=0.5, high=1.0), 0.0)

    def test_stable_inverse_score_is_monotonic(self):
        self.assertGreater(stable_inverse_score(0.0, 1.0), stable_inverse_score(1.0, 1.0))
        self.assertGreater(stable_inverse_score(1.0, 1.0), stable_inverse_score(2.0, 1.0))


if __name__ == "__main__":
    unittest.main()

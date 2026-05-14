import unittest

from scripts import split_bad_keys_by_reason as mod


class SplitBadKeysByReasonTests(unittest.TestCase):
    def test_training_unsafe_reasons_are_error_only(self):
        self.assertTrue(mod.is_error_reason("MissingOrInvalidFilesError"))
        self.assertTrue(mod.is_error_reason("NonFiniteDataError"))
        self.assertTrue(mod.is_error_reason("UnexpectedValueError"))

    def test_quality_reasons_remain_dirty_only(self):
        self.assertFalse(mod.is_error_reason("Rot6DInvalidError"))
        self.assertFalse(mod.is_error_reason("ExtremeStateActionDeltaError"))
        self.assertFalse(mod.is_error_reason("InstructionInvalidError"))


if __name__ == "__main__":
    unittest.main()

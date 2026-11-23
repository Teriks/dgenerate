import unittest

from dgenerate.cv2avnames import rewrite_avfoundation_class_names


class TestCv2AvNames(unittest.TestCase):

    def test_renames_both_classes_and_keeps_the_length(self):
        original = b'xx AVFFrameReceiver yy AVFAudioReceiver zz'
        updated = rewrite_avfoundation_class_names(original)
        self.assertEqual(len(updated), len(original))
        self.assertIn(b'CV2FrameReceiver', updated)
        self.assertIn(b'CV2AudioReceiver', updated)
        self.assertNotIn(b'AVFFrameReceiver', updated)
        self.assertNotIn(b'AVFAudioReceiver', updated)

    def test_second_pass_is_unchanged(self):
        once = rewrite_avfoundation_class_names(b'AVFFrameReceiver AVFAudioReceiver')
        self.assertEqual(rewrite_avfoundation_class_names(once), once)

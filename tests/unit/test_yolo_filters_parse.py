import unittest

import dgenerate.imageprocessors.util as _util


class TestYoloFiltersParse(unittest.TestCase):

    def test_class_filter_parsing(self):
        # Define a simple error function for testing
        def argument_error(msg):
            raise ValueError(msg)

        # Test with integer
        class_filter, index_filter = _util.yolo_filters_parse(0, None, argument_error)
        self.assertEqual(class_filter, {0})
        self.assertIsNone(index_filter)

        # Test with string
        class_filter, index_filter = _util.yolo_filters_parse('person', None, argument_error)
        self.assertEqual(class_filter, {'person'})
        self.assertIsNone(index_filter)

        # Test with list of integers
        class_filter, index_filter = _util.yolo_filters_parse([0, 1, 2], None, argument_error)
        self.assertEqual(class_filter, {0, 1, 2})
        self.assertIsNone(index_filter)

        # Test with list of strings
        class_filter, index_filter = _util.yolo_filters_parse(['person', 'dog', 'cat'], None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog', 'cat'})
        self.assertIsNone(index_filter)

        # Test with mixed list
        class_filter, index_filter = _util.yolo_filters_parse(['person', 0, 'dog'], None, argument_error)
        self.assertEqual(class_filter, {'person', 0, 'dog'})
        self.assertIsNone(index_filter)

        # Test with comma-separated string
        class_filter, index_filter = _util.yolo_filters_parse('person,dog,cat', None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog', 'cat'})
        self.assertIsNone(index_filter)

        # Test with mixed comma-separated string
        class_filter, index_filter = _util.yolo_filters_parse('person,dog,0', None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog', 0})
        self.assertIsNone(index_filter)

        # Test with quoted strings in comma-separated list
        class_filter, index_filter = _util.yolo_filters_parse('"person","dog","cat with spaces"', None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog', 'cat with spaces'})
        self.assertIsNone(index_filter)

        # Test with empty input
        class_filter, index_filter = _util.yolo_filters_parse(None, None, argument_error)
        self.assertIsNone(class_filter)
        self.assertIsNone(index_filter)

        # Test with empty list
        class_filter, index_filter = _util.yolo_filters_parse([], None, argument_error)
        self.assertEqual(class_filter, set())
        self.assertIsNone(index_filter)

        # Test with tuple
        class_filter, index_filter = _util.yolo_filters_parse(('person', 'dog'), None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog'})
        self.assertIsNone(index_filter)

        # Test with set
        class_filter, index_filter = _util.yolo_filters_parse({'person', 'dog'}, None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog'})
        self.assertIsNone(index_filter)

    def test_index_filter_parsing(self):
        # Define a simple error function for testing
        def argument_error(msg):
            raise ValueError(msg)

        # Test with integer
        class_filter, index_filter = _util.yolo_filters_parse(None, 0, argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, {0})

        # Test with list of integers
        class_filter, index_filter = _util.yolo_filters_parse(None, [0, 1, 2], argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, {0, 1, 2})

        # Test with empty input
        class_filter, index_filter = _util.yolo_filters_parse(None, None, argument_error)
        self.assertIsNone(class_filter)
        self.assertIsNone(index_filter)

        # Test with empty list
        class_filter, index_filter = _util.yolo_filters_parse(None, [], argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, set())

        # Test with tuple
        class_filter, index_filter = _util.yolo_filters_parse(None, (0, 1, 2), argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, {0, 1, 2})

        # Test with set
        class_filter, index_filter = _util.yolo_filters_parse(None, {0, 1, 2}, argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, {0, 1, 2})

        # Test with string integers
        class_filter, index_filter = _util.yolo_filters_parse(None, ["0", "1", "2"], argument_error)
        self.assertIsNone(class_filter)
        self.assertEqual(index_filter, {0, 1, 2})

    def test_index_filter_validation(self):
        # Test validation of index filter values
        error_messages = []

        def argument_error(msg):
            error_messages.append(msg)
            raise ValueError(msg)

        # Test with negative index
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(None, [-1, 0, 1], argument_error)
        self.assertIn('index-filter values must be greater than or equal to 0', error_messages[0])

        # Test with negative index as direct value
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(None, -1, argument_error)
        self.assertIn('index-filter values must be greater than or equal to 0', error_messages[0])

        # Test with negative index as string
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(None, ["-1", "0", "1"], argument_error)
        self.assertIn('index-filter values must be greater than or equal to 0', error_messages[0])

        # Test with non-integer value
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(None, ['a', 'b'], argument_error)
        self.assertIn('index-filter values must be integers', error_messages[0])

        # Test with non-iterable value
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(None, object(), argument_error)
        self.assertIn('index-filter values must be iterable', error_messages[0])

    def test_class_filter_negative_integer_validation(self):
        # Test validation of class filter negative integer values
        error_messages = []

        def argument_error(msg):
            error_messages.append(msg)
            raise ValueError(msg)

        # Test with negative integer in list
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse([-1, 0, 1], None, argument_error)
        self.assertIn('class-filter ID values must be greater than or equal to 0', error_messages[0])

        # Test with negative integer as direct value
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse(-1, None, argument_error)
        self.assertIn('class-filter ID values must be greater than or equal to 0', error_messages[0])

        # Test with negative integer as string in comma-separated list
        error_messages.clear()
        with self.assertRaises(ValueError):
            _util.yolo_filters_parse('person,dog,-1', None, argument_error)
        self.assertIn('class-filter ID values must be greater than or equal to 0', error_messages[0])

        # Test with negative integer as string in list - should NOT raise error (strings preserved)
        error_messages.clear()
        class_filter, index_filter = _util.yolo_filters_parse(['-1', '0', '1'], None, argument_error)
        self.assertEqual(class_filter, {'-1', '0', '1'})  # Strings should be preserved
        self.assertIsNone(index_filter)

    def test_combined_filters(self):
        # Define a simple error function for testing
        def argument_error(msg):
            raise ValueError(msg)

        # Test with both class and index filters
        class_filter, index_filter = _util.yolo_filters_parse([0, 1, 'person'], [0, 1, 2], argument_error)
        self.assertEqual(class_filter, {0, 1, 'person'})
        self.assertEqual(index_filter, {0, 1, 2})

        # Test with both class and index filters containing string integers - strings should be preserved
        class_filter, index_filter = _util.yolo_filters_parse(['0', '1', 'person'], ['0', '1', '2'], argument_error)
        self.assertEqual(class_filter, {'0', '1', 'person'})  # Strings should be preserved
        self.assertEqual(index_filter, {0, 1, 2})

    def test_class_filter_error_handling(self):
        error_messages = []

        def argument_error(msg):
            error_messages.append(msg)
            raise ValueError(msg)

        # Test with malformed comma-separated string that would cause an exception
        # Use an input that would cause tokenized_split to fail naturally
        with self.assertRaises(ValueError):
            # Unbalanced quotes should cause tokenized_split to fail
            _util.yolo_filters_parse('person,"dog,cat', None, argument_error)
        
        # Check that the error message contains the expected text
        self.assertTrue(any('Argument "class-filter"' in msg for msg in error_messages))

    def test_class_filter_preserves_numeric_strings(self):
        # Test that numeric strings are preserved as strings in class filter
        # This is important for model names that happen to be numeric strings
        def argument_error(msg):
            raise ValueError(msg)

        # Test with numeric string model names in direct input
        class_filter, index_filter = _util.yolo_filters_parse(['0', '42', 'person'], None, argument_error)
        self.assertEqual(class_filter, {'0', '42', 'person'})
        self.assertIsNone(index_filter)

        # Test with comma-separated string - these should be converted by filter_t
        class_filter, index_filter = _util.yolo_filters_parse('person,dog,0', None, argument_error)
        self.assertEqual(class_filter, {'person', 'dog', 0})  # 0 converted to int by filter_t
        self.assertIsNone(index_filter)


if __name__ == '__main__':
    unittest.main() 
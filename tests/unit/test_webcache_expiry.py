import datetime
import json
import os
import pathlib
import random
import sqlite3
import string
import tempfile
import time
import unittest

import dgenerate.filetypes as _filetypes
import dgenerate.mediainput as _mediainput
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.textprocessing as _textprocessing
from dgenerate.filecache import (
    DEFAULT_WEB_CACHE_EXPIRY,
    WebCacheExpiry,
    WebCacheExpiryParseError,
    WebFileCache,
    parse_partial_expiry_delta,
    _cache_name_extension_sql,
    _cache_value_extension_sql,
    _extension_from,
    _normalize_extension,
    _register_cache_sql,
    _stored_datetime,
)


def _backdate(cache, key, **delta):
    when = datetime.datetime.now() - datetime.timedelta(**delta)
    with cache.kv_store as kv:
        kv.cursor.execute(
            "UPDATE store SET creation_date=? WHERE key=?", (when, key))


class WebCacheExpiryParseTest(unittest.TestCase):
    def test_default_keeps_models_and_expires_everything_else_after_12_hours(self):
        policy = WebCacheExpiry.parse(DEFAULT_WEB_CACHE_EXPIRY)
        self.assertEqual(policy.delta_for('weights.safetensors'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('weights.gguf'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('photo.png'), datetime.timedelta(hours=12))
        self.assertEqual(policy.delta_for('clip.mp4'), datetime.timedelta(hours=12))
        self.assertEqual(policy.delta_for('notes.txt'), datetime.timedelta(hours=12))

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = WebFileCache(os.path.join(directory.name, 'cache.db'), directory.name)
        self.assertEqual(cache.expiry.delta_for('weights.ckpt'), policy.delta_for('weights.ckpt'))
        self.assertIsNone(cache.expiry_delta)

    def test_legacy_interval_uses_parse_timedelta(self):
        policy = WebCacheExpiry.parse('days=5;hours=6')
        self.assertTrue(policy.is_uniform)
        self.assertEqual(policy.uniform_delta, _textprocessing.parse_timedelta('days=5;hours=6'))
        self.assertEqual(policy.delta_for('weights.safetensors'), policy.uniform_delta)
        self.assertEqual(policy.delta_for('clip.mp4'), policy.uniform_delta)
        self.assertEqual(policy.delta_for('notes.txt'), policy.uniform_delta)

    def test_forever_and_empty_apply_to_every_file(self):
        for value in ('forever', '', None, '"forever"'):
            policy = WebCacheExpiry.parse(value)
            self.assertTrue(policy.is_uniform)
            self.assertEqual(policy.uniform_delta, datetime.timedelta.max)

    def test_rule_intervals_use_parse_timedelta(self):
        policy = WebCacheExpiry.parse(
            'models;forever, images;days=1;hours=6, videos;days=2, .png;hours=1, other;hours=12')
        self.assertFalse(policy.is_uniform)
        self.assertIsNone(policy.uniform_delta)
        self.assertEqual(policy.delta_for('model.safetensors'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.gguf'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.pt'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.pth'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.bin'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.ckpt'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.param'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.yaml'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('model.yml'), datetime.timedelta.max)
        self.assertEqual(
            policy.delta_for('photo.jpg'),
            _textprocessing.parse_timedelta('days=1;hours=6'))
        if 'mp4' in _filetypes.video_read_extension_set():
            self.assertEqual(policy.delta_for('clip.mp4'), datetime.timedelta(days=2))
        else:
            self.assertEqual(policy.delta_for('clip.mp4'), datetime.timedelta(hours=12))
        self.assertEqual(policy.delta_for('icon.png'), datetime.timedelta(hours=1))
        self.assertEqual(policy.delta_for('notes.txt'), datetime.timedelta(hours=12))

    def test_extension_overrides_category_and_unmatched_files_are_kept(self):
        policy = WebCacheExpiry.parse('images;days=1,.png;hours=1')
        self.assertEqual(policy.delta_for('icon.PNG'), datetime.timedelta(hours=1))
        self.assertEqual(policy.delta_for('photo.jpg'), datetime.timedelta(days=1))
        self.assertIsNone(policy.delta_for('notes.txt'))
        self.assertIsNone(policy.delta_for('weights.safetensors'))

    def test_filename_metadata_supplies_the_extension(self):
        policy = WebCacheExpiry.parse('models;forever,other;hours=12')
        self.assertEqual(
            policy.delta_for(
                os.path.join('cache', 'uuid'),
                {'filename': 'weights.safetensors'}),
            datetime.timedelta.max)
        self.assertEqual(
            policy.delta_for(os.path.join('cache', 'uuid.bin')),
            datetime.timedelta.max)

    def test_earliest_category_wins_when_sets_overlap(self):
        policy = WebCacheExpiry.parse('images;days=4,videos;days=9')
        overlap = _filetypes.image_read_extension_set() & _filetypes.video_read_extension_set()
        if not overlap:
            self.skipTest('this environment has no extension in both images and videos')
        ext = next(iter(overlap))
        self.assertEqual(policy.delta_for(f'file.{ext}'), datetime.timedelta(days=4))

    def test_later_duplicate_selector_replaces_the_earlier_one(self):
        policy = WebCacheExpiry.parse('.png;hours=1,.png;days=3')
        self.assertEqual(policy.delta_for('a.png'), datetime.timedelta(days=3))

    def test_singular_category_names_are_aliases(self):
        policy = WebCacheExpiry.parse('model;forever,image;days=1,video;days=2')
        self.assertEqual(policy.delta_for('a.gguf'), datetime.timedelta.max)
        self.assertEqual(policy.delta_for('a.png'), datetime.timedelta(days=1))
        if 'mp4' in _filetypes.video_read_extension_set():
            self.assertEqual(policy.delta_for('a.mp4'), datetime.timedelta(days=2))

    def test_partial_expiry_accepts_one_interval_and_rejects_a_rule_list(self):
        self.assertEqual(parse_partial_expiry_delta(), datetime.timedelta(hours=12))
        self.assertEqual(parse_partial_expiry_delta('days=2'), datetime.timedelta(days=2))
        self.assertEqual(
            parse_partial_expiry_delta('days=1;hours=6'),
            _textprocessing.parse_timedelta('days=1;hours=6'))
        self.assertEqual(parse_partial_expiry_delta('forever'), datetime.timedelta.max)
        self.assertEqual(parse_partial_expiry_delta(''), datetime.timedelta.max)
        with self.assertRaises(WebCacheExpiryParseError):
            parse_partial_expiry_delta('models;forever,other;hours=12')
        with self.assertRaises(WebCacheExpiryParseError):
            parse_partial_expiry_delta('.png;hours=1')
        with self.assertRaises(WebCacheExpiryParseError):
            parse_partial_expiry_delta('days=nope')

    def test_unknown_selector_and_bad_interval_raise(self):
        with self.assertRaises(WebCacheExpiryParseError):
            WebCacheExpiry.parse('imags;days=1')
        with self.assertRaises(WebCacheExpiryParseError):
            WebCacheExpiry.parse('.png;days=nope')
        with self.assertRaises(WebCacheExpiryParseError):
            WebCacheExpiry.parse('hours=12,days=1')
        with self.assertRaises(_textprocessing.TimeDeltaParseError):
            WebCacheExpiry.parse('not a delta')

    def test_category_sets_follow_the_formats_dgenerate_opens(self):
        self.assertEqual(
            set(_filetypes.image_read_extensions()),
            set(_mediainput.get_supported_image_formats()))
        self.assertEqual(
            _filetypes.video_read_extensions(),
            _mediainput._get_read_video_file_extensions())
        self.assertEqual(
            _filetypes.animation_read_extensions(),
            _mediainput.get_supported_animation_reader_formats())
        self.assertEqual(
            _filetypes.image_write_extensions(),
            _mediaoutput.get_supported_static_image_formats())
        self.assertEqual(
            _filetypes.animation_write_extensions(),
            _mediaoutput.get_supported_animation_writer_formats())
        self.assertEqual(
            _filetypes.tensor_extensions(),
            _mediainput.get_supported_tensor_formats())
        self.assertEqual(
            _filetypes.tensor_extensions(),
            _mediaoutput.get_supported_tensor_formats())
        self.assertIn('png', _filetypes.image_read_extension_set())
        self.assertIn('safetensors', _filetypes.model_extension_set())
        self.assertIn('gguf', _filetypes.model_extension_set())
        self.assertNotIn('cpkt', _filetypes.model_extension_set())
        self.assertEqual(_filetypes.video_write_extensions(), ['mp4'])
        self.assertTrue(_filetypes.tensor_extension_set() <= _filetypes.model_extension_set())
        self.assertIs(_filetypes.model_extension_set(), _filetypes.model_extension_set())
        self.assertEqual(_filetypes.image_read_extension_set(), frozenset(_filetypes.image_read_extensions()))
        if 'mp4' in _filetypes.video_read_extension_set():
            self.assertNotIn('mp4', _filetypes.image_read_extension_set())


class WebFileCacheExpiryTest(unittest.TestCase):
    def _cache(self, expression):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return WebFileCache(
            os.path.join(directory.name, 'cache.db'),
            directory.name,
            expiry_delta=expression)

    def _add(self, cache, key, ext, filename=None):
        metadata = {'mime-type': 'application/octet-stream'}
        if filename:
            metadata['filename'] = filename
        cached = cache.add(key, b'data', metadata=metadata, ext=ext)
        _backdate(cache, key, days=3)
        return cached

    def test_legacy_string_expires_every_file(self):
        cache = self._cache('days=1')
        self.assertEqual(cache.expiry_delta, datetime.timedelta(days=1))
        kept = cache.add('fresh', b'data', metadata={'filename': 'a.txt'}, ext='.txt')
        expired = self._add(cache, 'old', '.txt', 'a.txt')
        cache._clear_old_files()
        self.assertIsNotNone(cache.get('fresh'))
        self.assertIsNone(cache.get('old'))
        self.assertTrue(os.path.isfile(kept.path))
        self.assertFalse(os.path.isfile(expired.path))

    def test_rules_leave_unmatched_files_in_place(self):
        cache = self._cache('models;forever,images;days=1,.png;hours=1,other;hours=12')
        self.assertIsNone(cache.expiry_delta)
        model = self._add(cache, 'model', '.safetensors', 'weights.safetensors')
        png = self._add(cache, 'png', '.png', 'icon.png')
        jpg = self._add(cache, 'jpg', '.jpg', 'photo.jpg')
        other = self._add(cache, 'other', '.txt', 'notes.txt')
        # No rule matches .xyz once `other` is removed from consideration by using a
        # policy that does not name it. This file is covered by other and should go.
        cache._clear_old_files()
        self.assertIsNotNone(cache.get('model'))
        self.assertTrue(os.path.isfile(model.path))
        self.assertIsNone(cache.get('png'))
        self.assertFalse(os.path.isfile(png.path))
        self.assertIsNone(cache.get('jpg'))
        self.assertFalse(os.path.isfile(jpg.path))
        self.assertIsNone(cache.get('other'))
        self.assertFalse(os.path.isfile(other.path))

        kept = self._cache('models;forever,images;days=1')
        mystery = self._add(kept, 'mystery', '.xyz', 'blob.xyz')
        kept._clear_old_files()
        self.assertIsNotNone(kept.get('mystery'))
        self.assertTrue(os.path.isfile(mystery.path))

    def test_timedelta_argument_still_expires_every_old_file(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = WebFileCache(
            os.path.join(directory.name, 'cache.db'),
            directory.name,
            expiry_delta=datetime.timedelta(hours=12))
        self._add(cache, 'old', '.png', 'a.png')
        cache._clear_old_files()
        self.assertIsNone(cache.get('old'))

    def test_partials_follow_the_same_rules(self):
        cache = self._cache('models;forever,images;hours=1')
        partial_dir = pathlib.Path(cache.cache_dir, '.partials')
        partial_dir.mkdir()
        old = time.time() - datetime.timedelta(days=2).total_seconds()

        model = partial_dir / 'model-partial'
        model.write_bytes(b'model')
        (partial_dir / 'model-partial.meta').write_text(
            '{"metadata": {"filename": "weights.gguf"}, "ext": ".gguf"}',
            encoding='utf-8')
        image = partial_dir / 'image-partial'
        image.write_bytes(b'image')
        (partial_dir / 'image-partial.meta').write_text(
            '{"metadata": {"filename": "icon.png"}, "ext": ".png"}',
            encoding='utf-8')
        mystery = partial_dir / 'mystery-partial'
        mystery.write_bytes(b'mystery')
        (partial_dir / 'mystery-partial.meta').write_text(
            '{"metadata": {"filename": "blob.xyz"}, "ext": ".xyz"}',
            encoding='utf-8')
        broken = partial_dir / 'broken-partial'
        broken.write_bytes(b'broken')
        (partial_dir / 'broken-partial.meta').write_text(
            '{"metadata": "not-a-dict"}',
            encoding='utf-8')
        for path in partial_dir.iterdir():
            os.utime(path, (old, old))
        recent = partial_dir / 'recent-model-partial'
        recent.write_bytes(b'recent')
        (partial_dir / 'recent-model-partial.meta').write_text(
            '{"metadata": {"filename": "weights.gguf"}, "ext": ".gguf"}',
            encoding='utf-8')

        cache._clear_old_files()
        # forever applies to a finished model. The partial still expires.
        self.assertFalse(model.is_file())
        self.assertFalse((partial_dir / 'model-partial.meta').is_file())
        self.assertTrue(recent.is_file())
        self.assertTrue((partial_dir / 'recent-model-partial.meta').is_file())
        self.assertFalse(image.is_file())
        self.assertFalse((partial_dir / 'image-partial.meta').is_file())
        self.assertTrue(mystery.is_file())
        self.assertTrue((partial_dir / 'mystery-partial.meta').is_file())
        self.assertTrue(broken.is_file())
        self.assertTrue((partial_dir / 'broken-partial.meta').is_file())

    def test_partial_forever_cap_uses_partial_expiry_delta(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache = WebFileCache(
            os.path.join(directory.name, 'cache.db'),
            directory.name,
            expiry_delta='models;forever,images;hours=1',
            partial_expiry_delta='days=3')
        partial_dir = pathlib.Path(cache.cache_dir, '.partials')
        partial_dir.mkdir()
        two_days = time.time() - datetime.timedelta(days=2).total_seconds()
        four_days = time.time() - datetime.timedelta(days=4).total_seconds()

        kept = partial_dir / 'kept-model'
        kept.write_bytes(b'kept')
        (partial_dir / 'kept-model.meta').write_text(
            '{"metadata": {"filename": "weights.gguf"}, "ext": ".gguf"}',
            encoding='utf-8')
        expired = partial_dir / 'expired-model'
        expired.write_bytes(b'expired')
        (partial_dir / 'expired-model.meta').write_text(
            '{"metadata": {"filename": "weights.gguf"}, "ext": ".gguf"}',
            encoding='utf-8')
        image = partial_dir / 'image-partial'
        image.write_bytes(b'image')
        (partial_dir / 'image-partial.meta').write_text(
            '{"metadata": {"filename": "icon.png"}, "ext": ".png"}',
            encoding='utf-8')
        os.utime(kept, (two_days, two_days))
        os.utime(partial_dir / 'kept-model.meta', (two_days, two_days))
        os.utime(expired, (four_days, four_days))
        os.utime(partial_dir / 'expired-model.meta', (four_days, four_days))
        os.utime(image, (two_days, two_days))
        os.utime(partial_dir / 'image-partial.meta', (two_days, two_days))

        cache._clear_old_files()
        self.assertTrue(kept.is_file())
        self.assertTrue((partial_dir / 'kept-model.meta').is_file())
        self.assertFalse(expired.is_file())
        self.assertFalse((partial_dir / 'expired-model.meta').is_file())
        self.assertFalse(image.is_file())

    def test_sql_extension_matches_splitext(self):
        connection = sqlite3.connect(':memory:')
        _register_cache_sql(connection)
        name_sql = f"SELECT {_cache_name_extension_sql('name')} FROM (SELECT ? AS name)"
        value_sql = f"SELECT {_cache_value_extension_sql()} FROM (SELECT ? AS value)"

        def from_name(name):
            return connection.execute(name_sql, (name,)).fetchone()[0]

        samples = [
            '', '.', '..', '...', '.gitignore', '.bashrc.swp', '..png', 'file.',
            'file.png', 'file.png ', 'file.png\t', 'file.png\n', 'file.png\xa0',
            'file.png\u3000', 'file.png\v', 'a. png', 'file.tar.gz', 'PNG',
            'ab.ab', 'a.a.a', 'foo.bar.foo.bar.gz', 'abc/abc', 'abc/abc/abc',
            'aaa/aaa', 'a/a/a', 'file.txt/file.txt/file.txt',
            r'C:\Users\Eric.Name\.cache\web\uuid',
            r'C:\Users\Eric.Name\.cache\web\uuid.safetensors',
            '/home/eric.name/.cache/uuid',
            '/x/y/x/y/x/y', 'foo/bar/foo/bar/foo/bar.txt',
            "a'b.png", 'a"b.png', 'my file.png', 'фото.PNG', '模型.safetensors',
            'dir.with.dots/noext', 'dir.with.dots\\noext', 'foo.bar/baz.png',
            'foo.bar\\baz.png', 'a\nb.png', 'İ.PNG', 'weights.İ',
            '6f9c0e2a-1b3d-4c5e-8f70-aa11bb22cc33',
            '6f9c0e2a-1b3d-4c5e-8f70-aa11bb22cc33.safetensors',
        ]
        alphabet = string.ascii_letters + string.digits + " ._-'\"/\\."
        random.seed(7)
        for _ in range(2000):
            body = ''.join(random.choice(alphabet) for _ in range(random.randint(0, 50)))
            samples.append(body)
            samples.append(body + '/' + body)
            samples.append(body + '\\' + body + '.' + body)

        mismatches = []
        for name in samples:
            got = from_name(name)
            expect = _normalize_extension(os.path.splitext(str(name))[1])
            if got != expect:
                mismatches.append((name, got, expect))
                if len(mismatches) == 12:
                    break
        self.assertEqual(mismatches, [])

        entries = [
            {'path': r'C:\Users\Eric.Name\.cache\uuid', 'metadata': {}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid',
             'metadata': {'filename': 'weights.SAFETENSORS'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid.png',
             'metadata': {'filename': 'readme'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid.png',
             'metadata': {'filename': '.gitignore'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid',
             'metadata': {'filename': '..png'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid.png',
             'metadata': {'filename': '..png'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid',
             'metadata': {'filename': 'file.tar.gz'}},
            {'path': r'C:\Users\Eric.Name\.cache\uuid.png',
             'metadata': {'filename': r'C:\Users\Eric.Name\.cache\weights.safetensors'}},
            {'path': 'uuid.png', 'metadata': None},
            {'path': 'uuid.png', 'metadata': 'not-a-dict'},
            {'path': '/home/eric.name/.cache/uuid.PNG', 'metadata': {'filename': ''}},
            {'path': 'uuid', 'metadata': {'filename': "a'b\".png"}},
            {'path': 'uuid', 'metadata': {'filename': 'icon.PNG\u3000'}},
            {'path': 'uuid', 'metadata': {'filename': 'a. png'}},
        ]
        for entry in entries:
            got = connection.execute(value_sql, (json.dumps(entry),)).fetchone()[0]
            expect = _normalize_extension(_extension_from(entry.get('path'), entry.get('metadata')))
            self.assertEqual(got, expect, entry)

    def test_sql_delete_keeps_the_rows_python_would_keep(self):
        policy = (
            'models;forever,.png;hours=1,.tar.gz;hours=1,.gz;forever,'
            'images;days=4,videos;hours=1')
        cache = self._cache(policy)
        expiry = cache.expiry
        now = datetime.datetime.now()
        old = now - datetime.timedelta(days=5)
        mid = now - datetime.timedelta(days=2)
        overlap = _filetypes.image_read_extension_set() & _filetypes.video_read_extension_set()
        video_only = _filetypes.video_read_extension_set() - _filetypes.image_read_extension_set()

        def dotted(name):
            return os.path.join(cache.cache_dir, 'Eric.Name', '.cache', name)

        rows = []

        def add(key, path, metadata, created, present=True):
            if present and isinstance(path, str):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as handle:
                    handle.write(b'data')
            rows.append((key, path, metadata, created, present))
            value = json.dumps({'path': path, 'metadata': metadata})
            with cache.kv_store as kv:
                kv.cursor.execute(
                    "INSERT INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                    (key, value, created))

        add('bare', dotted('bare'), {}, old)
        add('model-path', dotted('model.safetensors'), {}, old)
        add('model-name', dotted('named'), {'filename': 'weights.SAFETENSORS'}, old)
        add('fallback-png', dotted('fallback.png'), {'filename': 'readme'}, mid)
        add('dotfile-fallback', dotted('dotfile.png'), {'filename': '.gitignore'}, mid)
        add('dotdot-png', dotted('dotdot'), {'filename': '..png'}, old)
        add('compound', dotted('compound'), {'filename': 'file.tar.gz'}, old)
        add('upper-png', dotted('upper'), {'filename': 'Icon.PNG'}, mid)
        add('quoted-png', dotted('quoted'), {'filename': "a'b\".png"}, mid)
        add('young-jpg', dotted('young.jpg'), {}, mid)
        add('old-jpg', dotted('old.jpg'), {}, old)
        add('fresh-png', dotted('fresh.png'), {'filename': 'fresh.png'}, now)
        add('space-png', dotted('space'), {'filename': 'file.png\u3000'}, mid)
        add('internal-space', dotted('internal'), {'filename': 'a. png'}, old)
        add('string-meta', dotted('meta.png'), 'not-a-dict', mid)
        add('null-meta', dotted('null.png'), None, mid)
        add('missing-file', dotted('gone.png'), {'filename': 'gone.png'}, old, present=False)

        if overlap:
            ext = sorted(overlap)[0]
            add('overlap', dotted(f'overlap.{ext}'), {}, mid)
        if video_only and 'mp4' in video_only:
            add('video', dotted('clip.mp4'), {}, mid)
        elif video_only:
            ext = sorted(video_only)[0]
            add('video', dotted(f'clip.{ext}'), {}, mid)

        # Readable as an entry, but the timestamp is not.
        for key, created in (
                ('empty-date', ''),
                ('garbage-date', 'not-a-date'),
                ('zero-date', '0'),
                ('null-date', None),
                ('date-only', '2020-01-01'),
                ('t-date', '2020-01-01T00:00:00'),
        ):
            add(key, dotted(f'{key}.png'), {'filename': f'{key}.png'}, created)

        # These would expire if the JSON were a cache entry. It is not.
        kept_corrupt = {
            'bad-json': 'not-json',
            'missing-metadata': json.dumps({'path': dotted('orphan.png')}),
            'missing-path': json.dumps({'metadata': {'filename': 'orphan.png'}}),
            'array-json': json.dumps(['uuid.png']),
            'null-value': None,
        }
        with cache.kv_store as kv:
            for key, value in kept_corrupt.items():
                kv.cursor.execute(
                    "INSERT INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                    (key, value, old))

        cache._clear_old_files()
        decided_at = datetime.datetime.now()

        for key, path, metadata, created, present in rows:
            parsed = _stored_datetime(created)
            meta = metadata if isinstance(metadata, dict) else None
            delta = expiry.delta_for(path, meta)
            try:
                cutoff = None if delta is None else decided_at - delta
            except OverflowError:
                cutoff = None
            expired = parsed is not None and cutoff is not None and parsed < cutoff
            if expired:
                self.assertIsNone(cache.get(key), key)
                if present:
                    self.assertFalse(os.path.isfile(path), key)
            else:
                self.assertIsNotNone(cache.get(key), key)
                if present:
                    self.assertTrue(os.path.isfile(path), key)

        for key in kept_corrupt:
            self.assertIn(key, cache.kv_store, key)

        other = self._cache('models;forever,other;hours=12')
        bare = os.path.join(other.cache_dir, 'Eric.Name', '.cache', 'uuid')
        model = os.path.join(other.cache_dir, 'Eric.Name', '.cache', 'weights.gguf')
        for path, payload in ((bare, b'data'), (model, b'model')):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as handle:
                handle.write(payload)
        with other.kv_store as kv:
            kv.cursor.execute(
                "INSERT INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                ('bare', json.dumps({'path': bare, 'metadata': {}}), old))
            kv.cursor.execute(
                "INSERT INTO store (key, value, creation_date) VALUES (?, ?, ?)",
                ('model', json.dumps({
                    'path': model,
                    'metadata': {'filename': 'weights.gguf'}}), old))
        reopened = WebFileCache(
            other.kv_store.db_path,
            other.cache_dir,
            expiry_delta='models;forever,other;hours=12')
        self.assertTrue(os.path.isfile(other.kv_store.db_path))
        self.assertIsNone(reopened.get('bare'))
        self.assertFalse(os.path.isfile(bare))
        self.assertIsNotNone(reopened.get('model'))
        self.assertTrue(os.path.isfile(model))


if __name__ == '__main__':
    unittest.main()

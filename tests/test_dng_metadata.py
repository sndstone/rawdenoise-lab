"""Standard metadata preservation without proprietary RAW fixtures."""
from pathlib import Path
import shutil
import struct

import numpy as np
import pytest
import rawpy
import tifffile

from denoise_lab.dng import RawMetadata, write_dng
from denoise_lab.tiff_metadata import read_metadata

ROOT = Path(__file__).resolve().parents[1]


def capture_tags():
    return {
        'root': {271: (2, 5, b'Test\0'), 272: (2, 7, b'Camera\0'),
                 306: (2, 20, b'2026:11:04 12:30:00\0'),
                 315: (2, 7, b'Artist\0'), 33432: (2, 6, b'Owner\0'),
                 50708: (2, 14, b'Test Camera 1\0'),
                 50722: (10, 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1)),
                 50779: (3, 1, (17,)), 50730: (10, 1, (-1, 2)),
                 50936: (2, 8, b'Profile\0')},
        'exif': {33434: (5, 1, (1, 125)), 33437: (5, 1, (28, 10)),
                 34855: (3, 1, (3200,)), 36867: (2, 20, b'2026:11:04 12:30:00\0'),
                 37386: (5, 1, (50, 1)), 42035: (2, 5, b'Test\0'),
                 42036: (2, 5, b'Lens\0'), 42034: (5, 4, (50, 1, 50, 1, 28, 10, 28, 10))},
        'gps': {0: (1, 4, (2, 3, 0, 0)), 1: (2, 2, b'N\0'),
                2: (5, 3, (52, 1, 22, 1, 30, 1)), 3: (2, 2, b'E\0'),
                4: (5, 3, (4, 1, 54, 1, 0, 1))}}


def test_exif_gps_and_colour_roundtrip(tmp_path):
    source, output = tmp_path/'source.dng', tmp_path/'output.dng'
    tags = capture_tags()
    meta = RawMetadata(source_tags=tags)
    codes = np.full((64, 80), 2000, np.uint16)
    write_dng(source, codes, meta, normalized=False)
    with rawpy.imread(str(source)) as raw:
        meta = RawMetadata.from_raw(raw, source)
        write_dng(output, raw.raw_image_visible.copy(), meta, normalized=False)
    result = read_metadata(output)
    for group, values in tags.items():
        for code, value in values.items():
            assert result[group][code] == value
    with tifffile.TiffFile(source) as a, tifffile.TiffFile(output) as b:
        tags_a, tags_b = a.pages[0].tags, b.pages[0].tags
        assert tags_a[50781].value != tags_b[50781].value
        assert tags_b[50827].value == 'source.dng'
        assert tags_b[305].value.startswith('rawdenoise ')
        assert 'denoised' in tags_b[270].value
        assert tags_b[34665].value['ISOSpeedRatings'] == 3200
        assert tags_b[34853].value['GPSLatitudeRef'] == 'N'
        thumbnail = b.pages[0].pages[0].asarray()
        assert thumbnail.dtype == np.uint8
        assert thumbnail.shape[-1] == 3
        assert max(thumbnail.shape[:2]) <= 256


@pytest.mark.parametrize('name', sorted(p.name for p in (ROOT/'examples/raw').glob('*.dng')))
def test_example_roundtrip(tmp_path, name):
    source = tmp_path/name
    shutil.copyfile(ROOT/'examples/raw'/name, source)
    output = tmp_path/'output.dng'
    with rawpy.imread(str(source)) as raw:
        meta = RawMetadata.from_raw(raw, source)
        codes = raw.raw_image_visible.copy()
        black, white = raw.black_level_per_channel, raw.white_level
        wb, matrix = raw.camera_whitebalance, raw.rgb_xyz_matrix.copy()
        write_dng(output, codes, meta, normalized=False)
    with rawpy.imread(str(output)) as raw:
        np.testing.assert_array_equal(raw.raw_image_visible, codes)
        np.testing.assert_array_equal(raw.black_level_per_channel, black)
        assert raw.white_level == white
        np.testing.assert_array_equal(raw.camera_whitebalance, wb)
        np.testing.assert_array_equal(raw.rgb_xyz_matrix, matrix)
        assert RawMetadata.from_raw(raw, output).phase == meta.phase
    a, b = read_metadata(source), read_metadata(output)
    for group, codes in [('root', (271, 272, 306, 50721)), ('exif', (33434, 34855, 36867))]:
        for code in codes:
            if code in a[group]:
                assert a[group][code] == b[group][code]


@pytest.mark.parametrize('endian', ['<', '>'])
@pytest.mark.parametrize('big', [False, True])
def test_ifd_walker_endian_subifd_cycle_and_bad_tag(tmp_path, endian, big):
    # IFD0 -> raw SubIFD -> IFD0 cycle, and a malformed exposure payload.
    width, entry_size, count_fmt, offset_fmt = (8, 20, 'Q', 'Q') if big else (4, 12, 'H', 'I')
    first = 16 if big else 8
    sub = first+struct.calcsize(count_fmt)+2*entry_size+width
    def entry(code, dtype, n, slot):
        return struct.pack(endian+'HH'+offset_fmt, code, dtype, n)+slot.ljust(width, b'\0')
    def offset(n):
        return struct.pack(endian+offset_fmt, n)
    header = (b'II' if endian == '<' else b'MM')+struct.pack(endian+'H', 43 if big else 42)
    header += struct.pack(endian+'HHQ', 8, 0, first) if big else offset(first)
    root = struct.pack(endian+count_fmt, 2)
    root += entry(330, 4, 1, struct.pack(endian+'I', sub))
    root += entry(33434, 5, 0xffffffff, offset(999999))
    root += offset(0)
    raw = struct.pack(endian+count_fmt, 2)
    raw += entry(271, 2, 4, b'Cam\0')
    raw += entry(34855, 3, 1, struct.pack(endian+'H', 800))
    raw += offset(first)
    path = tmp_path/'metadata.raw'
    path.write_bytes(header+root+raw)
    result = read_metadata(path)
    assert result['root'][271] == (2, 4, b'Cam\0')
    assert result['exif'][34855] == (3, 1, (800,))
    assert 33434 not in result['exif']


@pytest.mark.parametrize('data', [b'', b'not TIFF', b'II*\0\xff\xff\xff\xff', b'MM\0+\0\4\0\0'])
def test_bad_metadata_is_optional(tmp_path, data):
    path = tmp_path/'bad.raw'
    path.write_bytes(data)
    assert read_metadata(path) == {'root': {}, 'exif': {}, 'gps': {}}
    assert read_metadata(tmp_path/'missing.raw') == {'root': {}, 'exif': {}, 'gps': {}}


@pytest.mark.parametrize('unchanged', [False, True])
def test_geometry_opcodes_and_unsafe_tags(tmp_path, unchanged):
    # Include unsafe tags in a TIFF source, proving the reader rejects them.
    source = tmp_path/'source.tif'
    extratags = [(code, 7, 4, b'abcd', False) for code in (37500, 50781, 50931, 50932, 51008, 51009, 51022)]
    extratags += [(50829, 4, 4, (2, 4, 62, 76), False),
                  (50719, 4, 2, (4, 2), False), (50720, 4, 2, (72, 60), False)]
    tifffile.imwrite(source, np.zeros((64, 80), np.uint16), extratags=extratags)
    meta = RawMetadata(source_tags=read_metadata(source), geometry_unchanged=unchanged, visible_shape=(60, 72))
    output = tmp_path/'output.dng'
    write_dng(output, np.zeros((60, 72), np.uint16), meta, normalized=False, preview=False)
    with tifffile.TiffFile(output) as tif:
        tags = tif.pages[0].tags
        assert tags[50829].value == (0, 0, 60, 72)
        assert tags[50719].value == (0, 0)
        assert tags[50720].value == (72, 60)
        assert all(code not in tags for code in (37500, 50931, 50932, 51008))
        assert (51009 in tags) == unchanged
        assert (51022 in tags) == unchanged
        assert 330 not in tags


def test_minimal_legacy_bytes(tmp_path):
    # Frozen legacy tag values: this reference must not use the new writer's
    # tag construction, so changes to any legacy field fail the byte comparison.
    meta = RawMetadata(source_tags=capture_tags())
    codes = np.arange(64*80, dtype=np.uint16).reshape(64, 80)
    output, reference = tmp_path/'minimal.dng', tmp_path/'reference.dng'
    write_dng(output, codes, meta, normalized=False, metadata_mode='minimal')
    identity = (1000000, 1000000, 0, 1000000, 0, 1000000,
                0, 1000000, 1000000, 1000000, 0, 1000000,
                0, 1000000, 0, 1000000, 1000000, 1000000)
    legacy = [(254, 'I', 1, 0), (271, 's', 0, 'Clean-room'),
              (272, 's', 0, 'Bayer RAW'), (274, 'H', 1, 1),
              (33421, 'H', 2, (2, 2)), (33422, 'B', 4, (0, 1, 1, 2)),
              (50706, 'B', 4, (1, 4, 0, 0)), (50707, 'B', 4, (1, 1, 0, 0)),
              (50708, 's', 0, 'Clean-room Bayer RAW'), (50710, 'B', 3, (0, 1, 2)),
              (50711, 'H', 1, 1), (50713, 'H', 2, (2, 2)),
              (50714, '2I', 4, (0, 1000000)*4), (50717, 'I', 1, 65535),
              (50719, 'I', 2, (0, 0)), (50720, 'I', 2, (80, 64)),
              (50721, '2i', 9, identity), (50728, '2I', 3, (1000000, 1000000)*3),
              (50778, 'H', 1, 21)]
    tifffile.imwrite(reference, codes, photometric=32803, metadata=None,
                     extratags=[(*tag, False) for tag in legacy])
    assert output.read_bytes() == reference.read_bytes()


def test_raw_subifd_metadata_and_unicode_filename(tmp_path):
    source = tmp_path/'source.tif'
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.zeros((8, 8, 3), np.uint8), photometric='rgb', subifds=1,
                     extratags=[(271, 2, 0, b'Make\0', False)])
        writer.write(np.zeros((64, 80), np.uint16),
                     extratags=[(50721, 10, 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), False),
                                (50778, 3, 1, 21, False),
                                (33434, 5, 1, (1, 0), False)])
    metadata = read_metadata(source)
    assert metadata['root'][271] == (2, 5, b'Make\0')
    assert metadata['root'][50721][2][0:2] == (1, 1)
    assert 33434 not in metadata['exif']  # Reject invalid rationals.
    output = tmp_path/'output.dng'
    meta = RawMetadata(source_tags=metadata, source_name='caf\u00e9.raw')
    write_dng(output, np.zeros((64, 80), np.uint16), meta, normalized=False)
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].tags[50827].value == 'caf\u00e9.raw'
        # Pointer replacement must keep the TIFF entries numerically sorted.
        codes = [tag.code for tag in tif.pages[0].tags.values()]
        assert codes == sorted(codes)

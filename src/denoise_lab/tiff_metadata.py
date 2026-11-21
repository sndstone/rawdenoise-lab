"""Bounded, read-only classic TIFF/BigTIFF metadata extraction.

Only standard descriptive/colour tags are retained. Never follow maker notes,
image data offsets, embedded originals or signatures. Bad/unsupported IFDs are
ignored; decoding the sensor mosaic remains rawpy's responsibility.
"""
import struct
import math
from pathlib import Path

# TIFF data type -> struct format and scalar byte size (rationals have two scalars).
TYPES = {1: ('B', 1), 2: ('c', 1), 3: ('H', 2), 4: ('I', 4),
         5: ('I', 8), 6: ('b', 1), 7: ('B', 1), 8: ('h', 2),
         9: ('i', 4), 10: ('i', 8), 11: ('f', 4), 12: ('d', 8),
         13: ('I', 4), 16: ('Q', 8), 17: ('q', 8), 18: ('Q', 8)}
ROOT_TAGS = {271, 272, 306, 315, 33432, 33434, 33437, 34855, 36867, 36868,
             37386, 42034, 42035, 42036, 50706, 50707, 50708, 50709,
             50718, 50719, 50720, 50721, 50722, 50723, 50724, 50725,
             50726, 50727, 50728, 50729, 50730, 50731, 50732, 50733, 50734, 50736,
             50738, 50778, 50779, 50829, 50934,
             50936, 50937, 50938, 50939, 50940, 50941, 50942,
             50964, 50965, 51009, 51022, 51041, 51125}
EXIF_TAGS = {33434, 33437, 34850, 34855, 34864, 34865, 34866, 34867,
             36864, 36867, 36868, 36880, 36881, 36882, 37121, 37122,
             37377, 37378, 37379, 37380, 37381, 37382, 37383, 37384,
             37385, 37386, 37510, 37520, 37521, 37522, 40960, 40961,
             41483, 41486, 41487, 41488, 41495, 41728, 41729, 41730,
             41985, 41986, 41987, 41988, 41989, 41990, 41991, 41992,
             41993, 41994, 41996, 42032, 42033, 42034, 42035, 42036, 42037}


def read_metadata(path):
    """Return tag dictionaries for IFD0/raw, EXIF and GPS, without image reads.

    Limits: 64 IFDs, 4096 entries/IFD, 4 MiB/tag and 16 MiB total payload.
    Standard TIFF headers are required; opaque/non-TIFF RAW metadata is omitted.
    Values are (TIFF type, logical count, tuple of scalars or raw text/bytes).
    """
    groups = {'root': {}, 'exif': {}, 'gps': {}}
    try:
        with Path(path).open('rb') as f:
            size = f.seek(0, 2)
            def read(offset, count):
                if offset < 0 or count < 0 or offset + count > size:
                    raise ValueError('TIFF offset outside file')
                f.seek(offset)
                data = f.read(count)
                if len(data) != count:
                    raise ValueError('Truncated TIFF')
                return data
            header = read(0, 8)
            if header[:2] not in (b'II', b'MM'):
                return groups
            endian = '<' if header[:2] == b'II' else '>'
            magic = struct.unpack(endian+'H', header[2:4])[0]
            big = magic == 43
            if magic not in (42, 43):
                return groups
            if big and header[4:8] != struct.pack(endian+'HH', 8, 0):
                return groups
            width, entry_size, count_size = (8, 20, 8) if big else (4, 12, 2)
            offset_fmt = 'Q' if big else 'I'
            first = struct.unpack(endian+offset_fmt, read(8 if big else 4, width))[0]
            pending, seen, budget = [(first, 'root')], set(), 16*1024*1024
            while pending and len(seen) < 64:
                offset, group = pending.pop(0)
                if not offset or offset in seen:
                    continue
                seen.add(offset)
                try:
                    count = struct.unpack(endian+('Q' if big else 'H'), read(offset, count_size))[0]
                    if count > 4096:
                        continue
                    entries = read(offset+count_size, count*entry_size+width)
                    for i in range(count):
                        entry = entries[i*entry_size:(i+1)*entry_size]
                        code, dtype = struct.unpack(endian+'HH', entry[:4])
                        n = struct.unpack(endian+offset_fmt, entry[4:4+width])[0]
                        pointer = code in (330, 34665, 34853)
                        allowed = code in (ROOT_TAGS | EXIF_TAGS if group == 'root' else EXIF_TAGS if group == 'exif' else range(32))
                        if dtype not in TYPES or not (pointer or allowed):
                            continue
                        fmt, unit = TYPES[dtype]
                        length = n*unit
                        if not n or length > min(4*1024*1024, budget):
                            continue
                        slot = entry[4+width:4+2*width]
                        try:
                            data = slot[:length] if length <= width else read(struct.unpack(endian+offset_fmt, slot)[0], length)
                        except ValueError:
                            continue
                        budget -= length
                        value = data if dtype in (2, 7) else struct.unpack(endian+fmt*(length//struct.calcsize(fmt)), data)
                        if pointer:
                            if dtype in (4, 13, 16, 18) and n <= 64:
                                target = {330: 'root', 34665: 'exif', 34853: 'gps'}[code]
                                pending.extend((v, target) for v in value[:max(0, 64-len(seen)-len(pending))])
                        else:
                            if dtype in (5, 10) and any(v == 0 for v in value[1::2]):
                                continue
                            if dtype in (11, 12) and not all(math.isfinite(v) for v in value):
                                continue
                            if code in (271, 272) and dtype != 2:
                                continue
                            if code in (50721, 50722, 50723, 50724, 50964, 50965) and (dtype, n) != (10, 9):
                                continue
                            if code in (50778, 50779) and (dtype, n) != (3, 1):
                                continue
                            if code == 50728 and ((dtype, n) != (5, 3) or any(v <= 0 for v in value)):
                                continue
                            groups[group][code] = (dtype, n, value)
                    next_offset = struct.unpack(endian+offset_fmt, entries[-width:])[0]
                    if group == 'root' and len(seen)+len(pending) < 64:
                        pending.append((next_offset, group))
                except (ValueError, struct.error):
                    continue
    except (OSError, ValueError, struct.error):
        pass
    # Some TIFF-based RAWs put standard EXIF in IFD0 instead of an EXIF IFD.
    for code in EXIF_TAGS & groups['root'].keys():
        groups['exif'].setdefault(code, groups['root'].pop(code))
    return groups


def extratag(code, value):
    dtype, count, data = value
    return (code, dtype, count, data, False)


def append_ifds(path, groups):
    """Append standard EXIF/GPS IFDs and patch reserved LONG pointers.

    tifffile rejects EXIF/GPS extratags and its SubIFDs option creates image
    directories. Reserve private LONG tags, then replace their IDs and offsets
    after appending the standard directories. Output is classic little-endian TIFF.
    """
    import tifffile
    with tifffile.TiffFile(path) as tif:
        pointers = {name: (tif.pages[0].tags[code].offset, tif.pages[0].tags[code].valueoffset)
                    for name, code in [('exif', 34664), ('gps', 34852)] if groups[name]}
    with Path(path).open('r+b') as f:
        for name, (entry_offset, pointer) in pointers.items():
            f.seek(0, 2)
            if f.tell() % 2:
                f.write(b'\0')
            start = f.tell()
            items = sorted(groups[name].items())
            payload_start = start+2+12*len(items)+4
            entries, payload = bytearray(), bytearray()
            for code, (dtype, count, value) in items:
                fmt = TYPES[dtype][0]
                data = value if dtype in (2, 7) else struct.pack('<'+fmt*len(value), *value)
                if len(data) <= 4:
                    slot = data.ljust(4, b'\0')
                else:
                    slot = struct.pack('<I', payload_start+len(payload))
                    payload.extend(data)
                    if len(payload) % 2:
                        payload.append(0)
                entries.extend(struct.pack('<HHI', code, dtype, count)+slot)
            f.write(struct.pack('<H', len(items))+entries+b'\0'*4+payload)
            f.seek(entry_offset)
            f.write(struct.pack('<H', 34665 if name == 'exif' else 34853))
            f.seek(pointer)
            f.write(struct.pack('<I', start))

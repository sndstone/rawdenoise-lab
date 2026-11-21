"""Integer tile-grid geometry shared by NumPy and torch inference."""

def tile_starts(length, tile_size=512, overlap=58, grid_offset=0, grid_alignment=1):
    """Image-relative starts; positive offsets move the grid up/left.

    Alignment rounds overlap up (step down), so every origin, including the
    reflected apron, lies on the same network-stride lattice. Offsets wrap at
    the effective step and must already be multiples of the alignment.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) for n in
           (length, tile_size, overlap, grid_offset, grid_alignment)):
        raise ValueError("Tile geometry must use integers")
    if length < 1 or grid_alignment < 1 or tile_size < 1 or tile_size % grid_alignment:
        raise ValueError("Positive tile_size must be a multiple of grid_alignment")
    step = ((tile_size-overlap)//grid_alignment)*grid_alignment
    if not 0 <= overlap < tile_size or step < 1 or grid_offset % grid_alignment:
        raise ValueError("Invalid overlap or grid_offset not a multiple of grid_alignment")
    apron = tile_size-step
    first = -apron-grid_offset % step
    count = 1+max(0, (length+apron-first-tile_size+step-1)//step)
    return [first+i*step for i in range(count)]


def shift_offsets(tile_size, overlap, grid_alignment, count):
    """Evenly spread diagonal grid phases, quantized to the origin lattice."""
    tile_starts(1, tile_size, overlap, grid_alignment=grid_alignment)
    slots = (tile_size-overlap)//grid_alignment
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= slots:
        raise ValueError(f"tta_shift must be an integer in [1,{slots}]")
    return [i*slots//count*grid_alignment for i in range(count)]



"""FP32 fixture packing and checks; none of this runs inside a kernel harness."""
import numpy as np

from ipu_apps.kernel_registry.cases import PreparedCase


def random_values(shape):
    return np.random.default_rng(7).standard_normal(shape).astype('<f4')


def tile_input(values, *, halo=0, fill=0, guard=False):
    """CHW -> channel, padded row, halo tile, lane, including optional guard plane."""
    c, h, w = values.shape
    columns = 128 - 2 * halo
    tiles = (w + columns - 1) // columns
    packed = np.full((c + int(guard), h + 2 * halo, tiles, 128), fill, dtype='<f4')
    for tile in range(tiles):
        source = np.arange(128) + tile * columns - halo
        valid = (source >= 0) & (source < w)
        packed[:c, halo:halo+h, tile, valid] = values[:, :, source[valid]]
    return packed.reshape(-1, 128)


def untile_output(raw, shape, *, columns=128):
    c, h, w = shape
    tiles = (w + columns - 1) // columns
    return raw.reshape(c, h, tiles, 128)[..., :columns].reshape(c, h, -1)[..., :w]


def prepared_image(workspace, params, image, layout, check):
    inp, out = workspace / 'input.bin', workspace / 'output.bin'
    image = np.asarray(image, dtype='<f4').reshape(layout.input_rows, 128)
    inp.write_bytes(image.tobytes())

    def validate():
        raw = np.fromfile(out, dtype='<f4')
        if raw.size != layout.output_rows * 128:
            raise ValueError(f'output size mismatch: got {raw.size} FP32 elements')
        check(raw)

    return PreparedCase(params, {'input_path': inp, 'output_path': out}, validate)


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def tiled_case(workspace, params, layout, tiled, expected, *, columns=128, fill=0.0,
               constants=None):
    """Case whose image starts with ``tiled`` data and whose output is a tiled
    ``(C, H, W)`` result, checked against ``expected`` of that shape.

    Rows past the data hold ``fill``; ``constants(image)`` may write weights or
    other constant rows into them. ``columns`` is the number of real columns
    per output tile (``untile_output``).
    """
    image = np.full((layout.input_rows, 128), fill, dtype='<f4')
    image[:len(tiled)] = tiled
    if constants is not None:
        constants(image)
    return prepared_image(workspace, params, image, layout,
                          lambda raw: check_close(untile_output(raw, expected.shape, columns=columns),
                                                  expected))

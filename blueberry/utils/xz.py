import os
import lzma


def ensure_xzfile_decompressed(filename):
    if os.path.exists(filename):
        return

    xz_filename = filename + '.xz'
    print(f'** INFO: Dataset not decompressed, look for {xz_filename}...')
    if not os.path.exists(xz_filename):
        raise Exception(f'xz {xz_filename} not exists')

    print(f'** INFO: decompressing {xz_filename}...')
    text = lzma.open(xz_filename).read()
    with open(filename, 'wb') as f:
        f.write(text)

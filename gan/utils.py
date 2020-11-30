import os
import urllib.request
import tarfile

from config import DATA_URL

_ARCHIVE_NAME = "edges2shoes.tar.gz"


def _extract_tar(source_path, destination_path, compression_type='gz', delete_source=True):
    mode = 'r:{}'.format(compression_type if compression_type else 'r')
    with tarfile.open(source_path, mode) as fh:
        fh.extractall(path=destination_path)

    if delete_source:
        os.remove(source_path)


def download_data(url=DATA_URL, archive_name=_ARCHIVE_NAME):
    with urllib.request.urlopen(url) as response:
        data = response.read()
        with open(archive_name, 'wb') as archive:
            archive.write(data)

    _extract_tar(archive_name, '.')

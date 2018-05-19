"""Setup script for IJssel."""
import codecs
from distutils.core import setup
import os.path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    version_path = os.path.join(here, 'VERSION')

    with codecs.open(version_path, encoding='utf-8') as version_file:
        version = version_file.read().strip()

    setup(
        name='ijssel', version=version, modules=['ijssel'],
        url='https://github.com/jtv/ijssel',
        data_files=[(here, ['VERSION'])])


main()

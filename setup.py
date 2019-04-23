import numpy.distutils.misc_util as ndist_misc
from setuptools import Extension, setup

# semver with automatic minor bumps keyed to unix time
__version__ = '1.0.0'

setup(
    name="planet_helper",
    version=__version__,
    packages=["planet_helper"],
    data_files=[
        (
            'data', [
                'planet_helper/data/planet.tar.gz_00.part',
                'planet_helper/data/planet.tar.gz_01.part',
                'planet_helper/data/train_v2.csv.gz',
            ]
        ),
    ],
    include_package_data=True,
)

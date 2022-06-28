from setuptools import setup
from pathlib import Path

_dir = Path(__file__).resolve().parent

with open(f"{_dir}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open(f"{_dir}/nicks_plot_utils/__version__", "r", encoding="utf-8") as fh:
        version = fh.read()
        print(version)
except:
    print(_dir.name.split("-")[-1], flush=True)
    version = _dir.name.split("-")[-1]


setup(
    name='nicks_plot_utils',
    version=version,
    description='A example Python package',
    url='https://github.com/tylern4/nicks_plot_utils',
    author='Nick Tyler',
    author_email='nicholas.s.tyler.4@gmail.com',
    license='BSD 2-clause',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['nicks_plot_utils'],
    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],
    install_requires=['matplotlib',
                      'numpy',
                      'boost-histogram',
                      'scipy',
                      'lmfit',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.5',
)

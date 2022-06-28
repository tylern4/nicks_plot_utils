from setuptools import setup
import subprocess


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


git_version = subprocess.check_output(
    ['git', 'rev-list', '--count', 'HEAD']).decode("utf-8")[:-1]

with open("nicks_plot_utils/__version__", "r+", encoding="utf-8") as fh:
    template = fh.read()
    fh.seek(0)
    version_parts = list(map(int, template.split('.')))
    version_parts[-1] += 1
    version_parts = list(map(str, version_parts))
    version = ".".join(version_parts)
    fh.write(version)


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
                      'pandas'
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

from setuptools import setup

setup(
    name='nicks_plot_utils',
    version='1.0.0',
    description='A example Python package',
    url='https://github.com/tylern4/nicks_plot_utils',
    author='Nick Tyler',
    author_email='nicholas.s.tyler.4@gmail.com',
    license='BSD 2-clause',
    packages=['nicks_plot_utils'],
    install_requires=['matplotlib',
                      'numpy',
                      'boost-histogram',
                      'scipy',
                      'lmfit'
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
)

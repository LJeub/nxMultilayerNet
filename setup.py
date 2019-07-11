from setuptools import setup

setup(
    name='nxmultilayer',
    description='NetworkX based multilayer networks',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='Lucas G. S. Jeub',
    python_requires='>=3',
    install_requires=['networkx >= 2.3',
                      ],
    py_modules=['nxmultilayer'],
)

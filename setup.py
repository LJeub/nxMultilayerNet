from setuptools import setup

setup(
    name='nxMultilayerNet',
    description='NetworkX based multilayer networks',
    version='0.0.0',
    author='Lucas G. S. Jeub',
    python_requires='>=3',
    install_requires=['networkx >= 2.3',
                      ],
    py_modules=['nxmultilayer'],
)

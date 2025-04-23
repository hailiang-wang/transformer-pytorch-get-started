#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

LONGDOC = """
touchtext
---------------------

* PyPi https://pypi.org/project/touchtext/
* Sourcecode https://github.com/hailiang-wang/transformer-pytorch-get-started/tree/master/src/touchtext

A subset APIs replacement of torchtext, as torchtext is retired since 0.18.0 and only support pytorch 2.3.1.

```
pip install touchtext
```

# License
Copyright 2025, Hai Liang W., MIT License
Copyright 2024, Torchtext Team, BSD 3-Clause "New" or "Revised" License

"""

setup(
    name='touchtext',
    description='A subset APIs replacement of torchtext, as torchtext is retired since 0.18.0 and only support pytorch 2.3.1.',
    long_description=LONGDOC,
    version='0.0.2',
    author='Torchtext Team, Hai Liang W.',
    author_email='hailiang.hl.wang@gmail.com',
    url='https://github.com/hailiang-wang/transformer-pytorch-get-started/tree/master/src/touchtext',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.11',
        'Topic :: Utilities',
        'Development Status :: 5 - Production/Stable',
    ],
    license='MIT License',
    packages=['touchtext'],
    entry_points={
    },
    install_requires=[
        'torch >= 2.3.1',
        'tqdm',
    ],
)

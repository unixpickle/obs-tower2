from setuptools import setup

setup(
    name='obs-tower2',
    version='0.0.1',
    description='Obstacle tower private data',
    url='https://github.com/unixpickle/obs-tower2',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai reinforcement learning',
    packages=['obs_tower2'],
    install_requires=[
        'torch',
        'torchvision',
    ],
)

from setuptools import setup, find_packages

setup(
    name='ndf_rasterizer',
    version='0.1.1',
    packages=find_packages(),
    description='no design foundry – rasterizer plugin',
    long_description='Rasterizer plugin of no design foundry, it rasterizes a font and turns into a font.',
    author='Jan Sindler',
    author_email='jansindl3r@example.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='nodesignfoundry, rasterizer, plugin',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'freetype-py',
        'fonttools',
        'fontFeatures',
        'ufoLib2',
        'ufo-extractor'
    ],
    entry_points={
        'console_scripts': [
            'ndf_rasterizer=rasterizer.rasterizer:main',
        ],
    },
)
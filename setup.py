from setuptools import setup


setup(
    name='klabutils',
    version='0.0.0.0',
    description="A CLI and library for interacting with kesciLab training monitor service.",
    url='https://github.com/gerard0315/klabutils',
    packages=[
        'klabutils'
    ],
    package_dir={'klabutils': 'klabutils'},
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='klabutils',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
)

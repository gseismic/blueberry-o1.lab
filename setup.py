from setuptools import setup, find_packages


setup(
    name='blueberry', 
    version='0.0.1', 
    packages=find_packages(),
    description='blueberry-o1',
    install_requires = ['torch', 'numpy', 'loguru'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/blueberry',
    zip_safe=False,
    author_email='liushengli203@163.com'
)

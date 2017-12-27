from distutils.core import setup

dependencies = ['scipy','pandas','rasterio','numpy']
setup(
    name='NeilPy',
    version='0.1',
    packages=['neilpy',],
    license='MIT',
    long_description=open('README.txt').read(),
    url='http://github.com/Unidata/MetPy',

    author='Thomas Pingel',
    author_email='thomas.pingel@gmail.com',

    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Geographic Information Science',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: MIT License'],
    keywords='GIS lidar',
	install_requires=dependencies,
	
	)
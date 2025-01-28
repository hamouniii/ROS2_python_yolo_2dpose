from setuptools import find_packages, setup

package_name = 'vi_yolo_2dpose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='heidarshenas',
    maintainer_email='hamoun.heidarshenas@dfki.uni-kl.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_2dpose = vi_yolo_2dpose.yolo_2dpose_publisher:main',
        ],
    },
)

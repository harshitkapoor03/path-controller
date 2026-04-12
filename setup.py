from setuptools import setup

package_name = 'path_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Harshit',
    maintainer_email='harshit.kapoor1006@gmail.com',
    description='Path smoothing and trajectory tracking for differential drive robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_controller_node = path_controller.path_controller_node:main',
        ],
    },
)

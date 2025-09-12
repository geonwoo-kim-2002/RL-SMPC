from setuptools import find_packages, setup
import os

package_name = 'rl_switching_mpc'

def get_map_data_files():
    map_files = []
    for root, dirs, files in os.walk('maps'):
        for file in files:
            full_path = os.path.join(root, file)
            install_path = os.path.join('share', package_name, root)
            map_files.append((install_path, [full_path]))
    return map_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + get_map_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a',
    maintainer_email='apdnxn@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'service = rl_switching_mpc.server_test:main',
            'client = rl_switching_mpc.client_test:main',
            'run_gym = rl_switching_mpc.run_gym:main',
            'training = rl_switching_mpc.training:main',
        ],
    },
)

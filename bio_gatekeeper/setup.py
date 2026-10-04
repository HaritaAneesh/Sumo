from setuptools import find_packages, setup

package_name = 'bg_gatekeeper_pkg'

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
    maintainer='pavilion',
    maintainer_email='pavilion@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	  'bg_gatekeeper_node = bg_gatekeeper_pkg.bg_gatekeeper_node:main',
	  'rl_agent_node = bg_gatekeeper_pkg.rl_agent_node:main',
	  'traffic_env_node = bg_gatekeeper_pkg.traffic_env_node:main',
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="ecd_platform",
    version="0.2.0",
    description="Fault-tolerant ECD workflow platform with a PyQt6 desktop GUI for natural product AC determination",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
        "PyQt6>=6.5",
    ],
    extras_require={
        "build": ["pyinstaller>=6.6"],
    },
    entry_points={
        "console_scripts": [
            "ecd-platform=ecd_platform.cli:main",
        ],
        # GUI 为项目根目录下的脚本 gui_v1_2.py，直接运行: python gui_v1_2.py
    },
    python_requires=">=3.10,<3.15",
)

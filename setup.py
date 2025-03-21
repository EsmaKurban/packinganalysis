from setuptools import setup, find_packages

setup(
    author="Esma Kurban",
    description="A Python package for structural analysis of granular particle"
                "packings in 3D.",
    name="packinganalysis",
    version="0.1.0-dev",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "packinganalysis": ["data/*"],
    },
)

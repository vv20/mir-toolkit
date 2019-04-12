from setuptools import setup, find_packages

setup(
        name = "mirtoolkit",
        version = "0.1",
        description = "A library for music manipulation and analysis",
        author = "Victor Vasilyev",
        author_email = "victorvasilyev1806@gmail.com",
        modules = ["mirtoolkit"],
        packages = find_packages(),
        include_package_data = True,
        install_requires = [
            "pandas",
            "librosa",
            "mido",
            "numpy",
            "matplotlib",
            "mir_eval",
            "scikit-learn"
            ]
        )

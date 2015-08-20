from distutils.core import setup

setup(
    name = "cutiepy",
    packages = ["cutiepy"],
    package_data = {'cutiepy': ['include/*.h']},
    version = "0.0.1-dev",
    requires=['cython','numpy','scipy'],
    description = "Quantum Mechanics Toolkit",
    author = "Stefan Krastanov",
    author_email = "stefan@krastanov.org",
    url = "",
    download_url = "",
    keywords = ["phisics", "quantum"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description = """\
Quantum Mechanics Toolkit
=========================
"""
)

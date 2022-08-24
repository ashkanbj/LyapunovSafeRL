from setuptools import setup

setup(
    name="lyapunovrl",
    version="0.1",
    install_requires=[
        "cloudpickle==1.2.1",
        "gym",
        "ipython",
        "joblib",
        "matplotlib",
        "mpi4py",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "seaborn",
        "torch",
        "tqdm",
    ],
    description="A module",
    author="Ashkan BJ",
    packages=["lyapunovrl"],
)

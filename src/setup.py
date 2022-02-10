import setuptools

# Import the local code to set up.
import cadr  # noqa

NAME = "cadr"
DESCRIPTION = "Reinforcement learning library compatible with cadCAD"
MAINTAINER = "Anirudh A. Patel"
MAINTAINER_EMAIL = "anirudh@semiotic.ai"
VERSION = cadr.__version__
URL = "https://gitlab.semiotic.ai/cadcad-experiments/cadr"
LICENSE = "Apache 2.0"
PYTHON_REQUIRES = ">=3.9,<3.10"
MIN_DEPENDENCIES = [
    "numpy",
    "torch==1.10.2+cpu",
    "torchaudio==0.10.2+cpu",
    "torchvision==0.11.3+cpu",
]


def install_package():
    config = {
        "name": NAME,
        "description": DESCRIPTION,
        "maintainer": MAINTAINER,
        "maintainer_email": MAINTAINER_EMAIL,
        "version": VERSION,
        "packages": [NAME],
        "python_requires": PYTHON_REQUIRES,
        "install_requires": MIN_DEPENDENCIES,
    }

    setuptools.setup(**config)


if __name__ == "__main__":
    install_package()

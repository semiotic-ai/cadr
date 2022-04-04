# Copyright 2021 Semiotic AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

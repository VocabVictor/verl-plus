# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_file = 'verl/version.py'

def get_version():
    version_vars = {}
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(f.read(), version_vars)
    return version_vars['__version__']

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict<=0.6.2",
    "transformers",
    "wandb",
    "packaging>=20.0",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler", "torchvision", "qwen_vl_utils"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = ["tensordict<=0.6.2", "vllm<=0.8.5"]
SGLANG_REQUIRES = [
    "tensordict<=0.6.2",
    "sglang[srt,openai]==0.4.6.post5",
    "torch-memory-saver>=0.0.5",
    "torch==2.6.0",
]
TRL_REQUIRES = ["trl<=0.9.6"]
MCORE_REQUIRES = ["mbridge"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "mcore": MCORE_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=get_version(),
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["*.h", "*.cpp", "*.cu"],
        "verl": ["version.py", "llm/ds_config/*.json"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'verl=verl.cli.main:cli_main',
        ],
    },
)

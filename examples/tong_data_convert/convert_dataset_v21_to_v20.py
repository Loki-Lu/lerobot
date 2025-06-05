# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

##python3 examples/tong_data_convert/convert_dataset_v21_to_v20.py --repo-id Loki0929/so100_lan --root /home/tong/.cache/huggingface/lerobot/Loki0929/so100_lan --branch v2.0
## this code will update a new branch, !!!!! don't push to main !!!

import argparse

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, write_info, write_stats

V20 = "v2.0"
V21 = "v2.1"
CODEBASE_VERSION = V20


def convert_dataset(
    repo_id: str,
    new_repo_id: str,
    root: str,
    branch: str | None = None,
):
    # Load the original dataset
    dataset = LeRobotDataset(repo_id, revision=V21, root=root, force_cache_sync=False)

    # Remove old stats file if it exists
    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    # Write new stats
    write_stats(dataset.meta.stats, dataset.root)

    # Update codebase version in meta info
    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    # Delete episodes_stats.json file locally
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    # Initialize HfApi
    hub_api = HfApi()

    # Create a new dataset repository on Hugging Face Hub
    hub_api.create_repo(repo_id=new_repo_id, repo_type="dataset", exist_ok=True)

    # Push the dataset to the new repo_id
    dataset.push_to_hub(repo_id=new_repo_id, branch=branch, tag_version=False, allow_patterns="meta/")

    # Delete episodes_stats.json from the new repo if it exists
    if hub_api.file_exists(repo_id=new_repo_id, filename=EPISODES_STATS_PATH, revision=branch, repo_type="dataset"):
        hub_api.delete_file(
            path_in_repo=EPISODES_STATS_PATH,
            repo_id=new_repo_id,
            revision=branch,
            repo_type="dataset"
        )

    # Create a tag for the new repo
    hub_api.create_tag(repo_id=new_repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Original repository identifier on Hugging Face (e.g., `lerobot/pusht`).",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        required=True,
        help="New repository identifier on Hugging Face (e.g., `your_username/new_dataset`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Local directory to store the dataset.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))

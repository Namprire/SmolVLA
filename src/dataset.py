"""Load the required real Smol-LIBERO frame with current LeRobot.

The upstream dataset is published only in LeRobot v2.1 format. LeRobot 0.4.4
reads v3.0, so the smoke test creates a local, one-episode v3.0 view using the
official conversion utilities bundled with the installed package. The source
download is left untouched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from datasets import Dataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_info, write_episodes, write_info, write_stats
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_data,
    convert_info,
    convert_tasks,
    generate_episode_metadata_dict,
    legacy_load_episodes,
    legacy_load_episodes_stats,
)

DATASET_ID = "HuggingFaceVLA/smol-libero"
SMOKE_EPISODE = 0
PILOT_EPISODES = (0, 1, 2)
MAIN_EPISODES = tuple(range(10))


def _prepare_v21_source_view(raw_root: Path, view_root: Path, episodes: tuple[int, ...]) -> Path:
    """Make a lightweight converter input containing only selected parquet files."""
    if view_root.exists():
        shutil.rmtree(view_root)
    (view_root / "meta").mkdir(parents=True)
    (view_root / "data/chunk-000").mkdir(parents=True)

    for metadata_name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        shutil.copy2(raw_root / "meta" / metadata_name, view_root / "meta" / metadata_name)
    for episode_id in episodes:
        source = raw_root / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        target = view_root / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        target.symlink_to(source)
    return view_root


def prepare_smoke_dataset(raw_root: Path, converted_root: Path) -> Path:
    """Create an isolated v3.0 view containing official episode 0.

    This is deliberately scoped to the hard-gate smoke test. The later main
    experiment will need a separately documented multi-episode conversion.
    """
    raw_root = raw_root.resolve()
    converted_root = converted_root.resolve()

    source_episode = raw_root / "data/chunk-000/episode_000000.parquet"
    if not source_episode.is_file():
        raise FileNotFoundError(
            f"Missing {source_episode}. Run the README dataset download command first."
        )

    info_path = converted_root / "meta/info.json"
    if info_path.is_file():
        info = load_info(converted_root)
        if info.get("codebase_version") == "v3.0" and info.get("total_episodes") == 1:
            return converted_root

    staging_root = converted_root.with_name(f"{converted_root.name}.building")
    source_view = converted_root.with_name(f"{converted_root.name}.v21-source")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    _prepare_v21_source_view(raw_root, source_view, (SMOKE_EPISODE,))

    convert_info(source_view, staging_root, data_file_size_in_mb=100, video_file_size_in_mb=500)
    convert_tasks(source_view, staging_root)
    episode_file_metadata = convert_data(source_view, staging_root, data_file_size_in_mb=100)

    legacy_episodes = legacy_load_episodes(raw_root)
    legacy_episode_stats = legacy_load_episodes_stats(raw_root)
    selected_episodes = {SMOKE_EPISODE: legacy_episodes[SMOKE_EPISODE]}
    selected_stats = {SMOKE_EPISODE: legacy_episode_stats[SMOKE_EPISODE]}

    converted_episode_rows = list(
        generate_episode_metadata_dict(
            selected_episodes,
            episode_file_metadata,
            selected_stats,
            episodes_videos=None,
        )
    )
    write_episodes(Dataset.from_list(converted_episode_rows), staging_root)
    write_stats(aggregate_stats(list(selected_stats.values())), staging_root)

    info = load_info(staging_root)
    info["total_episodes"] = 1
    info["total_frames"] = selected_episodes[SMOKE_EPISODE]["length"]
    info["splits"] = {"train": "0:1"}
    write_info(info, staging_root)

    if converted_root.exists():
        shutil.rmtree(converted_root)
    staging_root.rename(converted_root)
    shutil.rmtree(source_view)
    return converted_root


def load_smoke_dataset(raw_root: Path, converted_root: Path) -> LeRobotDataset:
    """Return episode 0 through the installed LeRobotDataset v3.0 API."""
    root = prepare_smoke_dataset(raw_root, converted_root)
    return LeRobotDataset(
        repo_id=DATASET_ID,
        root=root,
        episodes=[SMOKE_EPISODE],
        revision="v3.0",
        download_videos=False,
    )


def prepare_pilot_dataset(
    raw_root: Path,
    converted_root: Path,
    episodes: tuple[int, ...] = PILOT_EPISODES,
) -> Path:
    """Create an isolated v3.0 view of a small contiguous episode subset.

    LeRobot's bundled v2.1 converter numbers source parquet files in sorted
    order, so this helper deliberately accepts only a zero-based contiguous
    prefix. This preserves the real episode identifiers without rewriting the
    source data.
    """
    raw_root = raw_root.resolve()
    converted_root = converted_root.resolve()
    episodes = tuple(episodes)
    if not episodes or episodes != tuple(range(len(episodes))):
        raise ValueError(
            "Pilot conversion requires a zero-based contiguous episode prefix; "
            f"got {episodes}"
        )

    for episode_id in episodes:
        source_episode = raw_root / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        if not source_episode.is_file():
            raise FileNotFoundError(
                f"Missing {source_episode}. Download the requested pilot episode first."
            )

    info_path = converted_root / "meta/info.json"
    if info_path.is_file():
        info = load_info(converted_root)
        if info.get("codebase_version") == "v3.0" and info.get("total_episodes") == len(episodes):
            return converted_root

    staging_root = converted_root.with_name(f"{converted_root.name}.building")
    source_view = converted_root.with_name(f"{converted_root.name}.v21-source")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    _prepare_v21_source_view(raw_root, source_view, episodes)

    # Keep the three ~34 MB episode files in one v3 data file. This also avoids
    # a file-index edge case in the bundled converter at its size boundary.
    convert_info(source_view, staging_root, data_file_size_in_mb=500, video_file_size_in_mb=500)
    convert_tasks(source_view, staging_root)
    episode_file_metadata = convert_data(source_view, staging_root, data_file_size_in_mb=500)

    if len(episode_file_metadata) != len(episodes):
        raise RuntimeError(
            "The pilot conversion saw an unexpected number of source parquet files: "
            f"expected {len(episodes)}, got {len(episode_file_metadata)}"
        )

    legacy_episodes = legacy_load_episodes(raw_root)
    legacy_episode_stats = legacy_load_episodes_stats(raw_root)
    selected_episodes = {episode_id: legacy_episodes[episode_id] for episode_id in episodes}
    selected_stats = {episode_id: legacy_episode_stats[episode_id] for episode_id in episodes}

    converted_episode_rows = list(
        generate_episode_metadata_dict(
            selected_episodes,
            episode_file_metadata,
            selected_stats,
            episodes_videos=None,
        )
    )
    write_episodes(Dataset.from_list(converted_episode_rows), staging_root)
    write_stats(aggregate_stats(list(selected_stats.values())), staging_root)

    info = load_info(staging_root)
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(selected_episodes[i]["length"] for i in episodes)
    info["splits"] = {"train": f"0:{len(episodes)}"}
    write_info(info, staging_root)

    if converted_root.exists():
        shutil.rmtree(converted_root)
    staging_root.rename(converted_root)
    shutil.rmtree(source_view)
    return converted_root


def load_pilot_dataset(
    raw_root: Path,
    converted_root: Path,
    episodes: tuple[int, ...] = PILOT_EPISODES,
) -> LeRobotDataset:
    """Return the locally converted real observations for the Phase 6 pilot."""
    root = prepare_pilot_dataset(raw_root, converted_root, episodes)
    return LeRobotDataset(
        repo_id=DATASET_ID,
        root=root,
        episodes=list(episodes),
        revision="v3.0",
        download_videos=False,
    )


def load_main_dataset(raw_root: Path, converted_root: Path) -> LeRobotDataset:
    """Return the isolated v3.0 view of episodes 0-9 for Phase 7."""
    root = prepare_pilot_dataset(raw_root, converted_root, MAIN_EPISODES)
    return LeRobotDataset(
        repo_id=DATASET_ID,
        root=root,
        episodes=list(MAIN_EPISODES),
        revision="v3.0",
        download_videos=False,
    )

from dataclasses import dataclass
from pathlib import Path


VALID_MAPPING_TYPES = {"human", "real_world"}

HUMAN_KEY_MAPPING = {
    'image': 'image',
    'depth': 'depth',
    'wrist_state': 'state/wrist',
    'hand_state': 'state/fingertips',
    'wrist_action': 'action/wrist',
    'hand_action': 'action/fingertips',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
    'presence': 'presence',
}

REAL_WORLD_KEY_MAPPING = {
    'image': 'image-head',
    'depth': 'depth-head',
    'wrist_state': 'state/wrist-head',
    'hand_state': 'state/fingertips-head',
    'wrist_action': 'action/wrist-head',
    'hand_action': 'action/fingertips-head',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic/head',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
}


@dataclass(frozen=True)
class ZarrListEntry:
    zarr_path: str
    mapping_type: str
    dataset_name: str
    wds_dataset: str


@dataclass(frozen=True)
class ParsedZarrListEntry:
    zarr_path: str
    mapping_type: str
    dataset_name: str
    target_name: str


def parse_zarr_list_entries(zarr_list_path, target_field_name):
    entries = []
    seen_dataset_names = {}

    with open(zarr_list_path) as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 1:
                zarr_path = parts[0]
                mapping_type = "human"
                target_name = None
            elif len(parts) == 2:
                zarr_path, mapping_type = parts
                if mapping_type not in VALID_MAPPING_TYPES:
                    raise ValueError(
                        f"{zarr_list_path}:{line_no}: expected '<zarr_path> [human|real_world] [{target_field_name}]'"
                    )
                target_name = None
            elif len(parts) == 3:
                zarr_path, mapping_type, target_name = parts
                if mapping_type not in VALID_MAPPING_TYPES:
                    raise ValueError(
                        f"{zarr_list_path}:{line_no}: mapping_type must be 'human' or 'real_world'"
                    )
            else:
                raise ValueError(
                    f"{zarr_list_path}:{line_no}: expected '<zarr_path> [human|real_world] [{target_field_name}]'"
                )

            if not Path(zarr_path).exists():
                raise ValueError(f"{zarr_list_path}:{line_no}: zarr path does not exist: {zarr_path}")

            dataset_name = Path(zarr_path).stem
            if dataset_name in seen_dataset_names:
                prev_line = seen_dataset_names[dataset_name]
                raise ValueError(
                    f"{zarr_list_path}:{line_no}: duplicate dataset name '{dataset_name}' (already used on line {prev_line})"
                )
            seen_dataset_names[dataset_name] = line_no

            if target_name is None:
                target_name = dataset_name
            if "/" in target_name or "\\" in target_name:
                raise ValueError(
                    f"{zarr_list_path}:{line_no}: {target_field_name} must be a simple directory name: {target_name}"
                )

            entries.append(ParsedZarrListEntry(
                zarr_path=zarr_path,
                mapping_type=mapping_type,
                dataset_name=dataset_name,
                target_name=target_name,
            ))

    if not entries:
        raise ValueError(f"{zarr_list_path}: no valid entries found")

    return entries


def parse_zarr_list(zarr_list_path):
    return [
        ZarrListEntry(
            zarr_path=entry.zarr_path,
            mapping_type=entry.mapping_type,
            dataset_name=entry.dataset_name,
            wds_dataset=entry.target_name,
        )
        for entry in parse_zarr_list_entries(zarr_list_path, "wds_dataset")
    ]

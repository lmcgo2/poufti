### INPUT / OUTPUT
import pandas as pd
import yaml
import argparse
import re
import sys
import h5py
import json
from pathlib import Path
from dataclasses import dataclass
from poufti.structures import Dataset, ExperimentData

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_pipeline_args() -> tuple[Path, Path]:
    """
    Parses command line arguments and returns (experiment_dir, config_path).
    """
    # Dynamically locate the project root based on where this Python file lives
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]  # Goes up src/ -> poufti/
    default_config_path = project_root / "configs" /"default_config.yaml"

    parser = argparse.ArgumentParser(description="Run the Poufti spatial cell tracking pipeline.")
    
    # Optional positional argument: Defaults to the current working directory (.)
    parser.add_argument(
        "data_dir", 
        type=Path, 
        nargs="?", 
        default=Path("."), 
        help="Path to the experiment folder. Defaults to the current directory."
    )
                        
    # Optional flag: Securely defaults to the root config.yaml
    parser.add_argument(
        "--config", 
        type=Path, 
        default=default_config_path, 
        help="Path to the YAML configuration file."
    )
    
    args = parser.parse_args()
    
    # Resolve paths to absolute paths immediately for downstream safety
    target_dir = args.data_dir.resolve()
    config_path = args.config.resolve()
    
    if not target_dir.is_dir():
        print(f"❌ Error: The directory '{target_dir}' does not exist.")
        sys.exit(1)
        
    if not config_path.is_file():
        print(f"❌ Error: The config file '{config_path}' does not exist.")
        sys.exit(1)
        
    return target_dir, config_path

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_experiment_folder(root_dir: Path) -> list[Dataset]:
    """
    Crawls the root directory and strictly pairs phXXX.nd2 files with trackingXXX.nd2 movies.
    """
    paired_datasets = []
    
    # 1. Find all phase ND2 files
    phase_files = sorted(root_dir.glob("ph*.nd2"))
    
    if not phase_files:
        print(f"⚠️ Warning: No phase ND2 files starting with 'ph' found in {root_dir.name}")
        return []

    for ph_file in phase_files:
        # 2. Extract the 3-digit ID (e.g., '001') using regex
        match = re.search(r'ph(\d+)', ph_file.stem)
        if not match:
            continue
            
        dataset_id = match.group(1)
        
        # 3. Construct the expected tracking movie path
        # Assuming the entire movie is now a single ND2 file, not a folder
        tracking_movie = root_dir / f"tracking{dataset_id}.nd2"
        
        # 4. Validation Checks
        if not tracking_movie.is_file():
            print(f"⚠️ Warning: Found {ph_file.name} but missing movie {tracking_movie.name}. Skipping.")
            continue
            
        # 5. Store as a clean Dataset object
        paired_datasets.append(Dataset(
            dataset_id=dataset_id,
            phase_image=ph_file,
            tracking_movie=tracking_movie  # Passing a single Path instead of a list
        ))
        
    print(f"✅ Successfully paired {len(paired_datasets)} datasets.")
    return paired_datasets

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

def save_dataset(dataset: list[ExperimentData], save_path: str, experiment_name: str):
    """
    Saves a list of ExperimentData objects (containing DataFrames) into a single HDF5 file.
    """
    # safely construct the path and force the .h5 extension
    file_name = f"{experiment_name}_dataset.h5"
    save_path = Path(save_path)
    full_path = save_path / file_name

    print(f"\nSaving dataset as {full_path}...")

    # open the file using pandas HDFStore
    # guarantees the file is safely closed when finished, even if an error occurs.
    with pd.HDFStore(full_path, mode='w') as store:
        
        # iterate through the dataset
        for idx, data in enumerate(dataset, start=1):
            
            # store cells and tracks data as subgroup for each data
            store.put(f'data_{idx:03d}/cells', data.cells, format='table', complib='zlib', complevel=5)
            store.put(f'data_{idx:03d}/tracks', data.tracks, format='table', complib='zlib', complevel=5)
            meta_json_string = json.dumps(data.metadata)
            meta_df = pd.DataFrame([{"experiment_metadata": meta_json_string}])
            store.put(f'data_{idx:03d}/metadata', meta_df, format='table', complib='zlib', complevel=5)
    print("Done")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
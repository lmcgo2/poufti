import yaml
import pandas as pd
import os
import nd2
import time

from pathlib import Path

from spotiflow.model import Spotiflow

from cellpose import models

from poufti.io import parse_pipeline_args, parse_experiment_folder, save_dataset

from poufti.mesh import split_boundary, build_internal_mesh

from poufti.segmentation import run_segmentation, create_cell_registry, create_polygon

from poufti.structures import CellData, Dataset

from poufti.tracking import run_spotiflow, run_laptrack, map_locus_to_cell, initialize_laptrack_object, enforce_track_length_minimum

from poufti.data_processing import track_dataframe_cleanup, compile_relational_data

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    start = time.time()
    print("\nStarting cell tracking pipeline...\n")

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]

    custom_models_path = project_root / "models" / "cellpose_models"

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(custom_models_path)
    
    # grab both paths from the terminal command
    experiment_dir, config_path = parse_pipeline_args()
    
    # load the YAML configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path.name}")
    
    # initialize cellpose model
    print(f"cellpose model: {config["cellpose_model"]}")
    cellpose_model = models.CellposeModel(gpu=True, model_type=config["cellpose_model"])

    # initialize spotiflow model
    print(f"spotiflow model: {config["spotiflow_model"]}")
    pretrained_path = "/home/lmcgo2/poufti/models/spotiflow_models/" + config["spotiflow_model"]
    spotiflow_model = Spotiflow.from_folder(
        pretrained_path=pretrained_path,
        inference_mode=True,
        map_location="cuda"
    )

    # initialize laptrack object
    lt = initialize_laptrack_object(**config["initialize_laptrack_object"])
    
    # parse the folder into logical Dataset pairs
    datasets = parse_experiment_folder(experiment_dir)

    if not datasets:
        print("Exiting pipeline. No valid datasets to process.")
        return

    processed_dataset = []
    
    # process each dataset
    for data in datasets:
        print(f"\n--- Processing Dataset {data.dataset_id} ---")
        
        print("Loading ND2 files into memory...")
        with nd2.ND2File(data.phase_image) as ph_file:
            phase_array = ph_file.asarray()
            if phase_array.ndim == 3 and phase_array.shape[0] == 1:
                phase_array = phase_array[0]
                
            extracted_metadata = {
                "pixel_size_x_um": ph_file.voxel_size().x,
                "pixel_size_y_um": ph_file.voxel_size().y,
                "axes_order": list(ph_file.sizes.keys()),
                "image_shape": dict(ph_file.sizes),
                # .text_info() contains Nikon's raw description fields
                "microscope_info": ph_file.text_info.get('description', 'None') 
            }
            
        with nd2.ND2File(data.tracking_movie) as trk_file:
            tracking_array = trk_file.asarray()
        print("Done.")
        # ---------------------------------------------------------

        print("Generating masks...")
        masks = run_segmentation(phase_array, cellpose_model, **config["run_segmentation"])
        print("Done.")

        print("Initializing cell registry...")
        LOCAL_CELL_REGISTRY = create_cell_registry(int(data.dataset_id), masks)
        print("Done.")

        print("Generating cell meshes...")
        for cell in LOCAL_CELL_REGISTRY.values():
            cell.local_polygon = create_polygon(cell, **config["create_polygon"])
            left_wall_points, right_wall_points = split_boundary(cell, **config["split_boundary"])
            cell.mesh_grid = build_internal_mesh(left_wall_points, right_wall_points, **config["build_internal_mesh"])
        print("Done.")

        print("Detecting loci...")
        spots_df = run_spotiflow(tracking_array, spotiflow_model, LOCAL_CELL_REGISTRY, **config["run_spotiflow"])
        print("Done.")

        print("Calculating tracks...")
        track_df = run_laptrack(lt, spots_df)
        print("Done.")

        print("Enforcing track length minimum...")
        track_df = enforce_track_length_minimum(track_df, config["minimum_track_length"])
        print("Done.")

        print("Converting coordinates to cellular position...")
        track_data = map_locus_to_cell(track_df, LOCAL_CELL_REGISTRY)
        print("Done.")

        print("Cleaning up dataset...")
        track_data = track_dataframe_cleanup(track_data)
        print("Done.")

        print("Calculating cell metrics...")
        processed_data = compile_relational_data(LOCAL_CELL_REGISTRY, track_data, extracted_metadata)
        print("Done.")
        
        processed_dataset.append(processed_data)
        
        print(f"--- Successfully Processed Dataset {data.dataset_id} ---")
        
    save_dataset(processed_dataset, config["save_path"], experiment_dir.name)
            
    print("\nAnalysis complete!")
    print(f"total duration: {time.time() - start} seconds\n")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
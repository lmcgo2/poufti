# **Poufti**
**Author: Liam McGoldrick<br>Kim Lab, University of Illinois at Urbana-Champaign**

High-throughput bacterial cell segmentation and loci tracking pipeline using Cellpose, Spotiflow, and Laptrack.

## **Config file parameters:**
- `image_prefix`: phase contrast images prefix (ex. phxxx.nd2 --> image_prefix = 'ph')
- `movie_prefix`: tracking movies prefix (ex. trackingxxx.nd2 --> movie_prefix = 'tracking')

- `cellpose_model`: cell segmentation model (I recommend using "bact_phase_cp3" for E. Coli, but you can look at the other pretrained models Cellpose offers, or train your own model)
- `spotiflow_model`: spot detection model (Defaults to "spotiflow_model_general", which is a pretrained model that I fine-tuned using mCherry and mVenus images using u-track as the ground-truth,but you can train or fine-tune your own model)

- `minimum_track_length`: minimum number of frames a track must be, else it is removed from the dataset

run_segmentation
- `diameter`: average width of cells in pixels (setting this to 0 makes cellpose determine the optimal setting itself. I tried measuring the diameter but I found using 0 is better)
- `cellprob_threshold`: decrease if cells are not being picked up, increase if model is picking up noise. [-6.0,6.0]
- `flow_threshold`: lower value makes cell outlines more strict and reduces cell merging, increasing allows for more variable shapes [0,1.0]
- `min_size`: minimum size a segmentation mask must be to be kept (in pixels)
- `remove_edge_masks`: removes cells with masks that are on the edge of the image

create_polygon
- `smoothing_proportionality_factor`: increase to make outlines smoother, decrease if outlines are being over-smoothed
- `num_of_intervals`: number of points used to make polygon outline

split_boundary:
- `num_longitudinal_points`: number of partitionings across the short axis of the cell (increasing improves subcellular position accuracy and vice versa)

build_internal_mesh
- `num_lateral_points`: number of partitionings across the long axis of the cell (increasing improves subcellular position accuracy and vice versa)

run_spotiflow
- `prob_thresh`: can be thought of as the minimum probability for suspected spot detection to be accepted

initialize_laptrack_object:
- `max_distance`: maximum distance (in pixels) that spots can be linked across frames

- `save_path`: path to where you want the processed dataset stored (example: "/home/lmcgo2/")

## **Setup (assuming in [ICRN](https://icrn.ncsa.illinois.edu/hub/login?next=%2Fhub%2F))**
1. Login to ICRN, and start session with:
   - environment: `Jupyter - PyTorch`
   - resource: `H200 141GB VRAM GPU, 10CPU/32GB`
2. Open the terminal, and run the following command to install Poufti: `pip install git+https://github.com/lmcgo2/poufti.git`
3. Using the command line, navigate to the poufti file (ex. `cd ~/poufti`)
4. In your command line, run `pip install poufti -e .`

## **Using poufti**
1. Using the command line, navigate to your experiment folder (ex. `cd ~/2609239-SK851`)
2. Type `poufti` in the command line, and hit enter
3. Wait for your results!

## **Advantages over traditional oufti & u-track pipeline**
1. Much easier to load dataset, no need to load every movie manually
2. This pipeline uses nd2 files, so no need to convert images or movies to tif files
3. Spots are assigned to cells before frame-linking to prevent spots from nearby cells being included in the same track (of course, this depends on the accuracy of the segmentation)
4. Runs much faster. I can process ~30 101 frame movies in about 6 minutes
5. In my opinion, it is much easier to understand the flow of this pipeline compared to oufti and u-track
6. Cellpose's cell segmentation appears to work very well for tightly packed cells
7. Probably some more that I am forgetting but this is it for now

## **Future improvements**
1. Jupyter notebook to test parameters
2. Jupyter notebook to plot analysis results

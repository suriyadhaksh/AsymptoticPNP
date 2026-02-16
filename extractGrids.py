import numpy as np
import pandas as pd
from pathlib import Path
import re

def extract_grids(trial_folder_path):
    """
    Extract epsilon, x_vec, t_vec, and solution grids (C1, C2, Phi)
    from a trial directory containing config.txt, data_init.plt, and output_manifest.csv.
    """

    trial_path = Path(trial_folder_path)
    print(f"Opening {trial_folder_path}")

    # ------------------------------------------------------------
    # 1. Read config.txt to extract epsilon (dbLength)
    # ------------------------------------------------------------
    config_path = trial_path / "config.txt"
    cfg_text = config_path.read_text()
    m = re.search(r"dbLength\s*=\s*([0-9.eE+-]+)\s*;", cfg_text)
    if not m:
        raise ValueError(f"dbLength not found in {config_path}")
    epsilon = float(m.group(1))

    # ------------------------------------------------------------
    # 2. Extract x_vec from data_init.plt
    # ------------------------------------------------------------
    data_init_path = trial_path / "data_initial.plt"
    with open(data_init_path, "r") as f:
        lines = f.readlines()

    # find ZONE line
    zone_line = next((i for i, l in enumerate(lines) if l.strip().startswith("ZONE")), None)
    if zone_line is None:
        raise ValueError("ZONE line not found in data_init.plt")

    # extract number of points I=...
    m = re.search(r"I\s*=\s*(\d+)", lines[zone_line])
    npts = int(m.group(1))
    start_idx = zone_line + 1
    end_idx = start_idx + npts
    data = np.loadtxt(lines[start_idx:end_idx])
    x_vec = data[:, 0]

    # ------------------------------------------------------------
    # 3. Extract t_vec and filenames from output_manifest.csv
    # ------------------------------------------------------------
    manifest_path = trial_path / "output_manifest.csv"
    df = pd.read_csv(manifest_path)
    t_vec = df["t"].to_numpy()
    file_names = df["file_name"].to_list()

    # ------------------------------------------------------------
    # 4. Initialize grids
    # ------------------------------------------------------------
    n_t = len(t_vec)
    n_x = len(x_vec)
    C1_grid = np.zeros((n_t, n_x))
    C2_grid = np.zeros((n_t, n_x))
    phi_grid = np.zeros((n_t, n_x))

    # ------------------------------------------------------------
    # 5. Fill grids from each .plt file
    # ------------------------------------------------------------
    for i, fname in enumerate(file_names):
        plt_path = trial_path / fname
        if not plt_path.exists():
            print(f"Warning: missing {fname}")
            continue

        with open(plt_path, "r") as f:
            lines = f.readlines()

        zone_line = next((j for j, l in enumerate(lines) if l.strip().startswith("ZONE")), None)
        if zone_line is None:
            raise ValueError(f"ZONE line not found in {fname}")

        m = re.search(r"I\s*=\s*(\d+)", lines[zone_line])
        npts = int(m.group(1))
        start_idx = zone_line + 1
        end_idx = start_idx + npts

        # Each line: x, C1, C2, Phi
        data = np.loadtxt(lines[start_idx:end_idx])
        C1_grid[i, :] = data[:, 1]
        C2_grid[i, :] = data[:, 2]
        phi_grid[i, :] = data[:, 3]

    # ------------------------------------------------------------
    # 6. Return all extracted arrays
    # ------------------------------------------------------------
    return epsilon, x_vec, t_vec, C1_grid, C2_grid, phi_grid



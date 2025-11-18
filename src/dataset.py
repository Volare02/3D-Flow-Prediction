import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, file_path, mode='train', spatial_crop_size = (32, 32, 32),
                 variables = ['UX_ms-1', 'UY_ms-1', 'UZ_ms-1', 'P_Pa'], num_spatial_crops = 10): 
        """
        Dataset for Variable Time-Step Prediction (Bidirectional).
        
        Args:
            file_path           (str)                   : Path to the HDF5 data file.
            mode                (str)                   : String represents the mode.
            spatial_crop_size   (Tuple[int, int, int])  : Dimensions (D, H, W) for spatial cropping.
            variables           (List[str])             : List of flow variables to load.
            num_spatial_crops   (int)                   : Number of deterministic spatial crops generated for each time pair.
        """
        super().__init__()

        self.file_path = file_path
        self.mode = mode
        self.spatial_crop_size = spatial_crop_size
        self.variables = variables
        self.num_spatial_crops = num_spatial_crops
        
        # HDF5 file handle for worker-local lazy loading.
        self.f = None 

        # Read Metadata.
        with h5py.File(file_path, 'r') as f: 
            sample_data = f["UX_ms-1/id_0000"][()]
            self.nx, self.ny, self.nz = sample_data.shape

            self.all_times = f["metadata/times"][:]
            self.total_snapshots = len(self.all_times)

        # Full Pairing.
        self.input_indices = list(range(0, self.total_snapshots)) 
        self.dt_steps = list(range(1 - self.total_snapshots, self.total_snapshots)) 
        self.valid_time_pairs = self._generate_valid_time_pairs()
        
        # Logging.
        T = self.total_snapshots
        print(f"FlowDataset initialized.")
        print(f"    Total Snapshots: {T}")
        print(f"    Total items    : {self.__len__()}")

    def __len__(self):
        """
        Returns the total logical size of the dataset.
        """
        return len(self.valid_time_pairs) * self.num_spatial_crops

    def __getitem__(self, index):
        """
        Loads the input state, target state, and time metadata (dt).
        """
        # 1. Determine Time Pair Indices (i, j).
        time_pair_index = index // self.num_spatial_crops 
        input_index, output_index = self.valid_time_pairs[time_pair_index]
        
        # 2. Calculate Time Difference.
        dt = self.all_times[output_index] - self.all_times[input_index]

        # 3. Spatial Cropping Coordinates.
        sx, sy, sz, dx, dy, dz = self._get_crop_coords_by_index(index)

        # 4. Read Data (Lazy I/O).
        f_handle = self._get_file_handle()
        input_list, target_list = [], []   
        for var in self.variables:
            key_in = f"{var}/id_{input_index:04d}"
            key_out = f"{var}/id_{output_index:04d}"
            
            input_data = f_handle[key_in][sx : sx + dx, sy : sy + dy, sz : sz + dz]
            target_data = f_handle[key_out][sx : sx + dx, sy : sy + dy, sz : sz + dz]
            
            input_list.append(input_data)
            target_list.append(target_data)

        # 5. Convert and Return.
        feats = torch.from_numpy(np.stack(input_list)).float()
        labels = torch.from_numpy(np.stack(target_list)).float()
        time_meta = torch.tensor(dt, dtype=torch.float32)
        
        return feats, labels, time_meta

    def _get_file_handle(self):
        """
        Opens HDF5 file if not already open (Worker-local lazy initialization).
        """
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')
        
        return self.f

    def _generate_valid_time_pairs(self):
        """
        Generates a list of all valid (input_index, output_index) pairs.
        """
        valid_pairs = []
        for input_index in self.input_indices: 
            for dt_step in self.dt_steps:
                output_index = input_index + dt_step
                if (output_index >= 0) and (output_index < self.total_snapshots):
                    valid_pairs.append((input_index, output_index))

        return valid_pairs

    def _get_crop_coords_by_index(self, index):
        """
        Generates deterministic spatial crop coordinates based on the index.
        """
        # Determine the unique ID for the time pair.
        time_pair_id = index // self.num_spatial_crops

        # Determine the specific crop ID.
        crop_id = index % self.num_spatial_crops

        # Use a seed for deterministic coordinate selection.
        seed_val = time_pair_id * 1000 + crop_id + (0 if self.mode == 'train' else 10000) 
        rng = random.Random(seed_val)
        
        crop_x, crop_y, crop_z = self.spatial_crop_size
        start_x = rng.randint(0, self.nx - crop_x)
        start_y = rng.randint(0, self.ny - crop_y)
        start_z = rng.randint(0, self.nz - crop_z)

        return start_x, start_y, start_z, crop_x, crop_y, crop_z
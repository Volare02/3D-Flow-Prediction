import h5py
import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 mode='train', 
                 spatial_size=(64, 64, 64), 
                 val_ratio=0.2, 
                 test_ratio=0.1, 
                 dt_per_frame=0.05):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            mode (str): 'train', 'val', or 'test'.
            spatial_size (tuple or int): Size of the spatial crop (D, H, W).
            dt_per_frame (float): Physical time interval per frame (e.g., 0.05s).
        """
        self.file_path = file_path
        self.mode = mode
        self.dt_per_frame = dt_per_frame
        
        # 1. Initialize file handle as None (Critical for Lazy Loading).
        self.f = None

        # Handle spatial_size input (convert int to tuple if necessary).
        if isinstance(spatial_size, int):
            self.crop_size = (spatial_size, spatial_size, spatial_size)
        else:
            self.crop_size = spatial_size

        # ====================================================
        # [CONFIG] Variable Names Mapping
        # ====================================================
        self.var_names = {
            'u': 'UX_ms-1',
            'v': 'UY_ms-1',
            'w': 'UZ_ms-1',
            'p': 'P_Pa'
        }

        self.samples = [] 
        
        # Open file ONCE just to build the index map, then close it immediately.
        with h5py.File(file_path, 'r') as f:
            # Check if variables exist.
            for v_key in self.var_names.values():
                if v_key not in f:
                    print(f"Warning: Key '{v_key}' not found in HDF5.")
            
            # Retrieve Time Steps (IDs) from the Pressure group.
            p_key = self.var_names['p']
            all_time_keys = sorted(list(f[p_key].keys()))
            total_frames = len(all_time_keys)
            
            # Split Dataset by Time (Sequential Split).
            n_test = int(total_frames * test_ratio)
            n_val = int(total_frames * val_ratio)
            n_train = total_frames - n_test - n_val
            
            if mode == 'train':
                self.time_ids = all_time_keys[:n_train]
            elif mode == 'val':
                self.time_ids = all_time_keys[n_train : n_train + n_val]
            elif mode == 'test':
                self.time_ids = all_time_keys[n_train + n_val:]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            print(f"[{mode.upper()}] Assigned {len(self.time_ids)} time frames.")
            
            # Generate All Samples (Grid Patches x Time Pairs).
            self._index_all_pairs(f, self.time_ids)

        print(f"[{mode.upper()}] Total samples generated: {len(self.samples)}")

    def _index_all_pairs(self, f_handle, time_keys):
        """
        Generate metadata for all valid (t_i, t_j) pairs where j >= i.
        """
        # 1. Get Domain Dimensions.
        first_id = time_keys[0]
        p_key = self.var_names['p']
        # Shape example: (1536, 384, 1024) -> (D, H, W).
        full_shape = f_handle[p_key][first_id].shape
        dim0, dim1, dim2 = full_shape
        
        cd, ch, cw = self.crop_size
        
        # 2. Define Spatial Grid (Non-overlapping -> Stride = Crop Size).
        s0_starts = range(0, dim0 - cd + 1, cd)
        s1_starts = range(0, dim1 - ch + 1, ch)
        s2_starts = range(0, dim2 - cw + 1, cw)
        
        # 3. Nested Loop for Time Pairs.
        num_times = len(time_keys)
        
        for i in range(num_times):
            for j in range(i, num_times): # Constraint: j >= i.
                t_curr = time_keys[i]
                t_next = time_keys[j]
                
                # Calculate time difference index.
                delta_idx = j - i 
                
                # Register all spatial crops for this time pair.
                for s0 in s0_starts:
                    for s1 in s1_starts:
                        for s2 in s2_starts:
                            self.samples.append({
                                't_curr': t_curr,
                                't_next': t_next,
                                'delta_idx': delta_idx,
                                's0': s0,
                                's1': s1,
                                's2': s2
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.f is None:
            # swmr=True allows reading even if the file is being written to (optional but robust)
            self.f = h5py.File(self.file_path, 'r', swmr=True)

        # 1. Retrieve Metadata.
        meta = self.samples[idx]
        s0, s1, s2 = meta['s0'], meta['s1'], meta['s2']
        cd, ch, cw = self.crop_size
        
        # 2. Read Data using persistent self.f handle.
        # NOTE: Do NOT close the file here, keep it open for the worker's lifetime.
        def read_crop(var_name, time_id):
            key = self.var_names[var_name]
            # Use self.f to read the specific slice.
            data = self.f[key][time_id][s0:s0+cd, s1:s1+ch, s2:s2+cw]
            return torch.from_numpy(data).float()

        # Input State (t_i).
        u_in = read_crop('u', meta['t_curr'])
        v_in = read_crop('v', meta['t_curr'])
        w_in = read_crop('w', meta['t_curr'])
        p_in = read_crop('p', meta['t_curr'])
        inputs = torch.stack([u_in, v_in, w_in, p_in], dim=0)
        
        # Target State (t_j).
        u_tar = read_crop('u', meta['t_next'])
        v_tar = read_crop('v', meta['t_next'])
        w_tar = read_crop('w', meta['t_next'])
        p_tar = read_crop('p', meta['t_next'])
        targets = torch.stack([u_tar, v_tar, w_tar, p_tar], dim=0)

        # 3. Calculate Physical Time Delta (dt).
        dt_val = meta['delta_idx'] * self.dt_per_frame
        dt_tensor = torch.tensor([dt_val], dtype=torch.float32)
        
        return inputs, targets, dt_tensor
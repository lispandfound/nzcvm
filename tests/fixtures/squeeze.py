import h5py
import numpy as np

def squeeze_hdf5_complete(input_file, output_file):
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:
        
        # 1. First, copy root attributes explicitly
        for attr_name, attr_value in src.attrs.items():
            dst.attrs[attr_name] = attr_value
        
        def process_item(name, obj):
            if isinstance(obj, h5py.Group):
                # Create group and copy its specific attributes
                new_group = dst.create_group(name)
                for attr_name, attr_value in obj.attrs.items():
                    new_group.attrs[attr_name] = attr_value
                    
            elif isinstance(obj, h5py.Dataset):
                # Squeeze the data
                data = obj[:]
                squeezed_data = np.squeeze(data)
                
                # Create dataset and copy its specific attributes
                # Note: chunks=True helps with performance for large arrays
                new_dset = dst.create_dataset(
                    name, 
                    data=squeezed_data, 
                    compression=obj.compression,
                    chunks=obj.chunks if squeezed_data.ndim > 0 else None
                )
                for attr_name, attr_value in obj.attrs.items():
                    new_dset.attrs[attr_name] = attr_value

        # 2. Walk the rest of the tree
        src.visititems(process_item)

# Usage
squeeze_hdf5_complete('./USGS_SFCVM_v21-0_detailed.berkeley.h5', './USGS_SFCVM_v21-0_detailed.berkeley.norm.h5')
print("Metadata and data successfully migrated.")

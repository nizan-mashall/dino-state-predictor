import h5py
import numpy as np

f = h5py.File('/code/data/demo.hdf5', 'r')
print("Keys:", list(f.keys()))
print("Number of demos:", len(f['data'].keys()))

demo = f['data']['demo_0']
print("Demo keys:", list(demo.keys()))
print("Number of steps:", demo['actions'].shape[0])
print("Action shape:", demo['actions'].shape)


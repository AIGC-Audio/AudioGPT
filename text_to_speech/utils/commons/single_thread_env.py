import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

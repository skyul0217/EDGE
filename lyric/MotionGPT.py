import numpy as np
import torch
import glob
import datetime

def extract_motion_dir(base_path):
	"""
	Returns a sorted list of names of batch samples directory.
 
	INPUT:
	- base_path: result batch directory of MotionGPT
				 ("samples_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND")
     
    OUTPUT:
    - motion_batch_dir: sorted list of names of batch samples directory
	"""
	def datedir_to_key(dirname):
		# Assume dirname consists of a form of "samples_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND"
		times = [int(time) for time in dirname.rstrip("/").split("_")[1].split("-")]
		key = datetime.datetime(*times).timestamp()
		return int(key)

	motion_batch_dir = sorted(glob.glob(f"{base_path}/samples*/"), key=lambda x:datedir_to_key(x))
	return motion_batch_dir

def extract_motion_batch(motion_batch_dir):
	"""
	Returns a list of names of batch samples directory.
 
	INPUT:
	- motion_batch_dir: sorted list of names of batch samples directory
 
	OUTPUT:
	- motion_batch: list of motion numpy arrays for each batch
					motion: (1, nframe, 22, 3)
	"""
	motion_batch = []
	for samples in motion_batch_dir:
		motions = []
		sample_list = sorted(glob.glob(f"./{samples}*_out.npy"), key=lambda x:int(x.split("_")[1].split("/")[-1]))
		for motion_npy in sample_list:
			motion = np.load(motion_npy)
			motions.append(motion)
		motion_batch.append(motions)
  
	return motion_batch

if __name__=="__main__":
    base_path = "./"
    batch_dir = extract_motion_dir(base_path)
    motion_batch = extract_motion_batch(batch_dir)
    for batch in motion_batch:
        for idx, motion in enumerate(batch):
            print(idx, motion.shape)
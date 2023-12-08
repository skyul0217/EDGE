import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

import lyric.lyric2motion as l2m
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

STRIDE = 2.5
HORIZON = 5.0

def test(opt):    
    torch.backends.cudnn.benchmark = False
    if opt.feature_type == "jukebox":
        feature_func = juke_extract
    elif opt.feature_type == "baseline":
        feature_func = baseline_extract
    else:
        raise Exception("Please select 'jukebox' or 'baseline' for feature_type")
    sample_length = opt.out_length
    sample_size = int(sample_length / STRIDE) - 1
    
<<<<<<< HEAD
    """
    lyric_list = []
=======
    # Lyric Domain
    print("LYRIC DOMAIN")
    all_lyric = []
>>>>>>> ae4bbe0 (Update 1208)
    if opt.lrc_path is not None:
        lrc_list = glob.glob(os.path.join(opt.lrc_path, "*.lrc"))
<<<<<<< HEAD
        for lrc in lrc_list:
            lyric = l2m.lyric_from_lrc(lrc)
            keyword = l2m.extract_keyword_period(lyric)
            transl = l2m.translate_lyric(keyword)
            lyric_list.append(transl)
    """
    
=======
        for lrc in tqdm(lrc_list):
            song_name = os.path.splitext(lrc)[0].split("/")[-1]
            song_file = os.path.join(opt.music_dir, song_name) + ".wav"
            
            # Extract translated keywords and timestamps
            print("[1/2] Extracting keywords:")
            lyric_lrc = l2m.lyric_from_lrc(lrc)
            keyword = l2m.extract_keyword_period(lyric_lrc)
            timestamps = l2m.translate_lyric(keyword)

            # Convert timestamp into horizon index
            print("[2/2] Converting into indices:")
            audio, sr = librosa.load(song_file, sr=None)
            start_idx = 0
            idx = 0
            window = int(HORIZON * sr)
            stride_step = int(STRIDE * sr)
            indices = list(range(0, len(audio), stride_step))
            idx_pairs = [(indices[i], indices[i+1]) for i, _ in enumerate(indices[:-1])]
            index_stamps = []
            for time_start, time_end, lyric_line in timestamps:
                time_start_idx, time_end_idx = int(time_start * sr), int(time_end * sr)
                window_start_idx, window_end_idx = -1, -1
                for idx, pair in enumerate(idx_pairs):
                    if pair[0] <= time_start_idx and time_start_idx < pair[1]:
                        window_start_idx = idx
                    
                    if pair[0] <= time_end_idx and time_end_idx < pair[1]:
                        window_end_idx = idx
                    
                    if window_start_idx >= 0 and window_end_idx >= 0:
                        break
                assert window_start_idx >=0 and window_end_idx >= 0
                index_stamps.append([window_start_idx, window_end_idx, lyric_line])
            all_lyric.append(index_stamps)
    
    # Music Domain
    print("\nMUSIC DOMAIN")
>>>>>>> ae4bbe0 (Update 1208)
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("[1/1] Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("[1/3] Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
                
            # slice the audio file
            print(f"[2/3] Slicing {wav_file}")
            slice_audio(wav_file, STRIDE, HORIZON, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            print(f"File List Size: {len(file_list)}")
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - sample_size)
            
            cond_list = []
            # generate juke representations
            print(f"[3/3] Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("\nMOTION DOMAIN")
    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i], all_lyric[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)

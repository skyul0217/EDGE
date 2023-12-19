import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import librosa
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
FPS = 30

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
    
    # Lyric Domain
    print("[LYRIC DOMAIN]")
    all_lyric = []
    if opt.lrc_path is not None:
        # Load all the "*.lrc" files
        lrc_list = glob.glob(os.path.join(opt.lrc_path, "*.lrc"))
        for lrc in tqdm(lrc_list):
            song_name = os.path.splitext(lrc)[0].split("/")[-1]
            song_file = os.path.join(opt.music_dir, song_name) + ".wav"
            
            # Extract translated keywords and timestamps
            print("[1/2] Extracting keywords:")
            lyric_lrc = l2m.lyric_from_lrc(lrc)
            keyword = l2m.extract_keyword_period(lyric_lrc)
            time_stamps = l2m.translate_lyric(keyword)

            # Convert timestamp into horizon index
            print("[2/2] Converting into indices:")
            for time_start, time_end, lyric_line in time_stamps:
                print(time_start, time_end)
            index_stamps = [[int(time_start * FPS), int(time_end * FPS), lyric_line] for time_start, time_end, lyric_line in time_stamps]
            generated_motions = l2m.generate_motion_lyric(song_name, index_stamps)
            print(generated_motions[0].shape)
            stamp_song = [[song_name, idx_start, idx_start + int(motion.shape[1]), lyric, motion] for (idx_start, idx_end, lyric), motion in zip(index_stamps, generated_motions)]
            """
            # Adjust for the sample idx
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
                index_stamps.append({f"{song_name}": [window_start_idx, window_end_idx, lyric_line]})
            """
            all_lyric.append(stamp_song)
        print([lyric[:-1] for lyric in all_lyric])
    
    # Music Domain
    print("\n[MUSIC DOMAIN]")
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
            # rand_idx = random.randint(0, len(file_list) - sample_size)
            # Test for lyric combination
            rand_idx = 30
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("[1/3] Computing features for input music")
        for i, wav_file in enumerate(glob.glob(os.path.join(opt.music_dir, "*.wav"))):
            songname = os.path.splitext(os.path.basename(wav_file))[0]
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
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
            # randomly sample a chunk of length at most sample_size
            # rand_idx = random.randint(0, len(file_list) - sample_size)
            # -> Test for lyric combination
            rand_idx = 30

            idx_render_start = int(rand_idx * FPS * STRIDE)
            idx_render_end = int((rand_idx + sample_size) * FPS * STRIDE)
            print(f"Slicing between {idx_render_start} and {idx_render_end}")

            print("Before")
            print([comp[:-1] for comp in all_lyric[i]])

            for key_bar in all_lyric[i]:
                print(f"{int(key_bar[1])} vs {idx_render_start} / {int(key_bar[2])} vs {idx_render_end}")
                if int(key_bar[2]) <= idx_render_start or int(key_bar[1]) >= idx_render_end:
                    key_bar[1] = -1
                    key_bar[2] = -1
                elif int(key_bar[1]) <= idx_render_start and int(key_bar[2]) >= idx_render_end:
                    key_bar[1] = 0
                    key_bar[2] = int(sample_size * FPS * STRIDE)
                elif int(key_bar[1]) <= idx_render_start and int(key_bar[2]) <= idx_render_end:
                    key_bar[1] = 0
                    key_bar[2] -= idx_render_start
                elif int(key_bar[2]) >= idx_render_end and int(key_bar[1]) >= idx_render_start:
                    key_bar[1] -= idx_render_start
                    key_bar[2] = int(sample_size * FPS * STRIDE)
                else:
                    key_bar[1] -= idx_render_start
                    key_bar[2] -= idx_render_start
                print(f"-> {int(key_bar[1])} ({int(key_bar[1]) / (FPS)}) / {int(key_bar[2])} ({int(key_bar[2]) / (FPS)})")

            print("After")
            print([comp[:-1] for comp in all_lyric[i]])
            
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

    print("\n[MOTION DOMAIN]")
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

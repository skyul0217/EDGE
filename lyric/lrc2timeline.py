import argparse
import numpy as np

def lyric_with_timeline(lrc_file):
    with open(lrc_file, "rb") as f:
        lrc_data = f.readlines()
    
    lrc_data = [line.decode("utf-8") for line in lrc_data]
    
    lrc_dict = dict()
    times = []
    lyrics = []

    for line in lrc_data:
        time, lyric = line.lstrip('[').split(']')
        try:
            time = int(time.split(':')[0]) * 60 + float(time.split(':')[1])
        except Exception:
            continue
        times.append(time)
        lyrics.append(lyric.strip('\n'))
    
    lrc_dict["timeline"] = np.array(times)
    lrc_dict["lyric"] = np.array(lyrics)

    return lrc_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="./result", required=False)
    args = parser.parse_args()

    print(lyric_with_timeline(args.lrc_file))
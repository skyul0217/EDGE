# Common
from tqdm import tqdm

# Import for Keyword Extraction
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

# Import for Korean-English Translation
from transformers import MarianMTModel, MarianTokenizer

# Import for Text-to-Motion Generation
# import MotionGPT

# Import LRC file
def lyric_from_lrc(lrc_file, is_with_timeline=True):
    with open(lrc_file, "rb") as f:
        lrc_data = f.readlines()
    
    lrc_data = [line.decode("utf-8") for line in lrc_data if line.strip() != ""]
    
    times = []
    lyrics = []

    print("Extracting lyrics from LRC file...")
    for idx in tqdm(range(len(lrc_data))):
        time, lyric = lrc_data[idx].lstrip('[').split(']')
        if idx < len(lrc_data) - 1:
            next_time, _ = lrc_data[idx+1].lstrip('[').split(']')
        else:
            next_time = -1.0
        try:
            time = round(int(time.split(':')[0]) * 60 + float(time.split(':')[1]), 2)
            if next_time != -1:
                next_time = round(int(next_time.split(':')[0]) * 60 + float(next_time.split(':')[1]), 2)
            times.append([time, next_time])
            lyrics.append(lyric.strip('\n'))
        except Exception:
            continue
    
    times = np.array(times)
    lyrics = np.array(lyrics)

    if is_with_timeline:
        lrc_result = np.array([[time[0], time[1], lyric] for time, lyric in zip(times, lyrics)])
    else:
        lrc_result = lyrics

    return lrc_result

def extract_keyword_period(lyrics, song_info=None, topk=10, is_with_timeline=True):
    # XLM-R 모델과 토크나이저 초기화
    MODEL_NAME = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaModel.from_pretrained(MODEL_NAME)

    # lyric.txt 파일 읽기
    """with open('lyric_test.txt', 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines() if line.strip()]"""
    moments = None
    if is_with_timeline:
        moments = lyrics[:, 0:2]
        original_sentences = lyrics[:, 2]
    else:
        original_sentences = lyrics

    if song_info is None:
        # 각 문장에 대한 BERT 임베딩 생성
        sentence_embeddings = []
        print("Extracting keywords...")
        for sentence in tqdm(original_sentences):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 문장의 CLS 토큰 임베딩 사용
            sentence_embeddings.append(outputs.last_hidden_state[0][0].numpy())

        # 코사인 유사도를 사용하여 문장 간 유사성 행렬 생성
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # TextRank 그래프 생성
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # 가장 중요한 문장 순위 매기기
        ranked_sentences_with_score = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)

        # 상위 10개 문장 출력
        topk_ranked_sentences = []
        kth_rank = 0
        while len(topk_ranked_sentences) < topk:
            if ranked_sentences_with_score[kth_rank][1] not in topk_ranked_sentences:
                topk_ranked_sentences.append(ranked_sentences_with_score[kth_rank][1])
            kth_rank += 1

    # lyric_info.txt 파일 읽기
    else:
        with open('lyric_info.txt', 'r', encoding='utf-8') as file:
            words = [word.split()]

        # 각 문장에 대한 BERT 임베딩 생성
        word_embeddings = []
        for word in word_embeddings:
            inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 문장의 CLS 토큰 임베딩 사용
            word_embeddings.append(outputs.last_hidden_state[0][0].numpy())

        # 코사인 유사도를 사용하여 문장 간 유사성 행렬 생성
        similarity_matrix_word = cosine_similarity(word_embeddings)

        # TextRank 그래프 생성
        nx_graph_word = nx.from_numpy_array(similarity_matrix_word)
        scores_words = nx.pagerank(nx_graph_word)

        # 가장 중요한 문장 순위 매기기
        ranked_words = sorted(((scores_words[i], s) for i, s in enumerate(words)), reverse=True)

        # 상위 10개 문장 출력
        for i in range(10):
            print(ranked_words[i][1])
    
    if moments is None:
        keyword_result = np.array(topk_ranked_sentences)
    else:
        topk_ranked_moments = []
        for sentence in topk_ranked_sentences:
            idx = np.where(original_sentences == sentence)[0]
            topk_ranked_moments.append(moments[idx])
        topk_ranked_moments_nodup = []
        for indices in topk_ranked_moments:
            for moment in indices:
                topk_ranked_moments_nodup.append(moment)
        
        keyword_periods = []
        for k in range(topk):
            keyword_with_time = [[moment[0], moment[1], topk_ranked_sentences[k]] for moment in topk_ranked_moments[k]]
            keyword_periods.extend(keyword_with_time)

        keyword_result = np.array(keyword_periods)
    
    return keyword_result


def translate_lyric(lyrics, is_with_timeline=True):
    transl_model_name = 'Helsinki-NLP/opus-mt-ko-en'
    transl_tokenizer = MarianTokenizer.from_pretrained(transl_model_name)
    transl_model = MarianMTModel.from_pretrained(transl_model_name)

    # 점수와 함께 저장된 문장만 추출
    if is_with_timeline:
        moments = lyrics[:, 0:2].astype(float).tolist()
        sentences = lyrics[:, 2]
    else:
        sentences = lyrics

    # 문장을 토크나이즈하고 번역
    translated_lyrics = []
    
    print("Translating keywords...")
    for sentence in tqdm(sentences):
        input_tokenized = transl_tokenizer(sentence, return_tensors="pt")
        output_generated = transl_model.generate(**input_tokenized)
        translated_text = transl_tokenizer.decode(output_generated[0], skip_special_tokens=True)
        translated_lyrics.append(translated_text)

    if is_with_timeline:
        translated_result = [[moment[0], moment[1], lyric] for moment, lyric in zip(moments, translated_lyrics)]
    else:
        translated_result = translated_lyrics

    translated_result = sorted(translated_result, key=lambda x: x[0])
        
    return translated_result

def generate_motion_lyric(translation, is_with_timeline=True):
    if is_with_timeline:
        pass
    else:
        pass
    return None

if __name__=="__main__":
    lyrics = lyric_from_lrc("./NewJeans_ETA.lrc")
    keywords = extract_keyword_period(lyrics, topk=10)
    translation = translate_lyric(keywords)
    for i in translation:
        print(f"{i[0]:.2f} - {i[1]:.2f}: {i[2]}")
    # motion = generate_motion_lyric(translation)
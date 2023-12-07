import langid
from transformers import MarianMTModel, MarianTokenizer

# 한국어-영어 번역기
model_name_ko_en = 'Helsinki-NLP/opus-mt-ko-en'
tokenizer_ko_en = MarianTokenizer.from_pretrained(model_name_ko_en)
model_ko_en = MarianMTModel.from_pretrained(model_name_ko_en)

# 영어-한국어 번역기
model_name_en_ko = 'Helsinki-NLP/opus-mt-tc-big-en-ko'
tokenizer_en_ko = MarianTokenizer.from_pretrained(model_name_en_ko)
model_en_ko = MarianMTModel.from_pretrained(model_name_en_ko)

input_text = "서둘러서 정리해 걔는 real bad"

# 문장을 단어 단위로 분리하고 각 단어의 언어를 판별
words = input_text.split()
langs = [langid.classify(word)[0] for word in words]

# 영어 단어만 한국어로 번역
english_words = []
translated_words = []
for word, lang in zip(words, langs):
    if lang == 'en':
        model_inputs = tokenizer_en_ko([word], return_tensors="pt")
        output_generated = model_en_ko.generate(**model_inputs)
        translated_word = tokenizer_en_ko.decode(output_generated[0], skip_special_tokens=True)
        print(word, translated_word)
        translated_words.append(translated_word)
    else:
        translated_words.append(word)

# 번역된 단어들을 다시 하나의 문장으로 합침
translated_text_ko = ' '.join(translated_words)
print(translated_text_ko)

# 한국어 문장을 영어로 번역
model_inputs = tokenizer_ko_en([translated_text_ko], return_tensors="pt")
output_generated = model_ko_en.generate(**model_inputs)
translated_text_en = tokenizer_ko_en.decode(output_generated[0], skip_special_tokens=True)

print(translated_text_en)
import requests
from bs4 import BeautifulSoup

title = input("title: ")
artist = input("artist: ")

url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=' + title + artist + '가사'

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    lyrics = soup.select_one('#main_pack > div.sc_new.cs_common_module._au_music_content_wrap.case_empasis.color_15 > div.cm_content_wrap > div.cm_content_area._cm_content_area_song_lyric')
    if lyrics == None:
        print("찾으시는 곡의 가사 정보를 찾을 수 없습니다.")
    else:
        lines = lyrics.select('p')
        
        # 파일명 설정
        filename = f"{title}_{artist}.txt"

        # 텍스트 파일로 저장
        with open(filename, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line.get_text() + "\n")
        print(f"가사가 저장된 파일: {filename}")

else : 
    print(response.status_code)
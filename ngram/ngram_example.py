def generate_ngrams(text, n):
    """
    주어진 텍스트에서 n-gram을 생성합니다.
    
    Args:
        text (str): 처리할 입력 텍스트
        n (int): 각 n-gram의 길이
        
    Returns:
        list: n-gram 목록
    """
    # 소문자로 변환하고 단어로 분리
    words = text.lower().split()
    
    # n-gram 생성
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def character_ngrams(text, n):
    """
    주어진 텍스트에서 문자 단위 n-gram을 생성합니다.
    
    Args:
        text (str): 처리할 입력 텍스트
        n (int): 각 n-gram의 길이
        
    Returns:
        list: 문자 단위 n-gram 목록
    """
    text = text.lower()
    ngrams = []
    
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    
    return ngrams

# 사용 예시
if __name__ == "__main__":
    sample_text = "자연어 처리는 언어학과 인공지능의 한 분야입니다."
    
    print("샘플 텍스트:")
    print(sample_text)
    print()
    
    # 단어 단위 n-gram
    print("단어 단위 n-gram:")
    print("유니그램 (n=1):", generate_ngrams(sample_text, 1))
    print("바이그램 (n=2):", generate_ngrams(sample_text, 2))
    print("트라이그램 (n=3):", generate_ngrams(sample_text, 3))
    
    print()
    
    # 문자 단위 n-gram
    print("문자 단위 n-gram:")
    sample_word = "자연어"
    print(f"'{sample_word}'의 문자 바이그램:", character_ngrams(sample_word, 2))
    print(f"'{sample_word}'의 문자 트라이그램:", character_ngrams(sample_word, 3))
    
    # 실용적인 응용 예시 - 공통 n-gram 찾기
    text1 = "빠른 갈색 여우가 게으른 개를 뛰어넘습니다"
    text2 = "빠른 갈색 여우가 어제 게으른 개를 뛰어넘었습니다"
    
    print("\n두 텍스트 간의 공통 n-gram 찾기:")
    ngrams1 = set(generate_ngrams(text1, 3))
    ngrams2 = set(generate_ngrams(text2, 3))
    common_ngrams = ngrams1.intersection(ngrams2)
    
    print("텍스트 1:", text1)
    print("텍스트 2:", text2)
    print("공통 트라이그램:", common_ngrams if common_ngrams else "없음") 
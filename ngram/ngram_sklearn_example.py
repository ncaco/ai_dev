import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# 샘플 데이터: 영화 리뷰 (긍정/부정 감정)
reviews = [
    # 긍정적인 리뷰
    "이 영화는 환상적이었어요! 정말 즐겁게 봤습니다.",
    "훌륭한 연기와 놀라운 스토리라인. 강력 추천합니다.",
    "올해 본 영화 중 최고의 작품 중 하나입니다.",
    "감독이 이 걸작을 정말 훌륭하게 만들었습니다.",
    "뛰어난 연기와 함께하는 멋진 영화 경험이었습니다.",
    "처음부터 끝까지 재미있어서 눈을 뗄 수 없었습니다.",
    "탁월한 캐릭터들이 있는 멋진 영화입니다.",
    "웃고 울었으며, 정말 감정적인 여정이었습니다.",
    # 부정적인 리뷰
    "이 영화는 끔찍했어요, 완전히 시간 낭비였습니다.",
    "연기가 형편없고 지루한 줄거리. 추천하지 않습니다.",
    "올해 본 영화 중 최악의 작품 중 하나입니다.",
    "감독이 흥미로운 것을 전혀 만들어내지 못했습니다.",
    "실망스러운 경험과 함께 평범한 연기였습니다.",
    "처음부터 끝까지 지루해서 잠이 들었습니다.",
    "캐릭터가 제대로 발전되지 않은 최악의 영화입니다.",
    "내내 짜증났고, 정말 불쾌한 경험이었습니다."
]

# 레이블 (1: 긍정, 0: 부정)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.25, random_state=42
)

print("scikit-learn을 이용한 n-gram 텍스트 분석\n")
print("샘플 크기:", len(reviews))
print("학습 샘플:", len(X_train))
print("테스트 샘플:", len(X_test))

# 예제 1: 단어 유니그램 사용
print("\n1. 단어 유니그램 (Bag of Words)")
print("-" * 50)

# CountVectorizer와 나이브 베이즈를 사용하는 파이프라인 생성
unigram_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 1))),  # 유니그램
    ('classifier', MultinomialNB())
])

# 학습 및 평가
unigram_pipeline.fit(X_train, y_train)
y_pred = unigram_pipeline.predict(X_test)

print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 예제 2: 단어 유니그램과 바이그램 사용
print("\n2. 단어 유니그램 및 바이그램")
print("-" * 50)

# CountVectorizer(유니그램 + 바이그램)와 나이브 베이즈를 사용하는 파이프라인 생성
bigram_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # 유니그램 및 바이그램
    ('classifier', MultinomialNB())
])

# 학습 및 평가
bigram_pipeline.fit(X_train, y_train)
y_pred = bigram_pipeline.predict(X_test)

print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 예제 3: 문자 n-gram
print("\n3. 문자 n-gram (n=3 ~ 5)")
print("-" * 50)

# 문자 n-gram과 나이브 베이즈를 사용하는 파이프라인 생성
char_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        analyzer='char',
        ngram_range=(3, 5)  # 문자 트라이그램, 4-그램, 5-그램
    )),
    ('classifier', MultinomialNB())
])

# 학습 및 평가
char_pipeline.fit(X_train, y_train)
y_pred = char_pipeline.predict(X_test)

print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 예제 4: 단어 n-gram을 사용한 TF-IDF
print("\n4. 단어 n-gram을 사용한 TF-IDF (n=1 ~ 2)")
print("-" * 50)

# TF-IDF와 나이브 베이즈를 사용하는 파이프라인 생성
tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        ngram_range=(1, 2),
        use_idf=True,
        smooth_idf=True
    )),
    ('classifier', MultinomialNB())
])

# 학습 및 평가
tfidf_pipeline.fit(X_train, y_train)
y_pred = tfidf_pipeline.predict(X_test)

print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 가장 정보가 풍부한 특징(TF-IDF 값이 가장 높은 n-gram) 표시
print("\n가장 중요한 n-gram (TF-IDF 기준):")
vectorizer = tfidf_pipeline.named_steps['vectorizer']
feature_names = vectorizer.get_feature_names_out()

# 분류기 계수 가져오기 - feature_log_prob_ 사용
classifier = tfidf_pipeline.named_steps['classifier']
# 클래스 0(부정)에 대한 로그 확률과 클래스 1(긍정)에 대한 로그 확률의 차이
feature_importance = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

# 중요도 계수 값별로 특징 정렬
top_positive_idx = np.argsort(feature_importance)[-10:]  # 상위 10개 긍정 특징
top_negative_idx = np.argsort(feature_importance)[:10]   # 상위 10개 부정 특징

print("\n상위 긍정 n-gram:")
for idx in top_positive_idx[::-1]:
    print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

print("\n상위 부정 n-gram:")
for idx in top_negative_idx:
    print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

# 새로운 리뷰로 시도
print("\n새로운 리뷰 분류:")
new_reviews = [
    "이 영화를 정말 좋아했어요, 정말 놀라웠어요!",
    "완전 재앙이었어요, 이 영화는 완전히 실망스러웠습니다."
]

for review in new_reviews:
    print(f"\n리뷰: {review}")
    
    # 각 모델로 예측
    unigram_pred = unigram_pipeline.predict([review])[0]
    bigram_pred = bigram_pipeline.predict([review])[0]
    char_pred = char_pipeline.predict([review])[0]
    tfidf_pred = tfidf_pipeline.predict([review])[0]
    
    print(f"유니그램 모델 예측: {'긍정' if unigram_pred == 1 else '부정'}")
    print(f"바이그램 모델 예측: {'긍정' if bigram_pred == 1 else '부정'}")
    print(f"문자 n-gram 모델 예측: {'긍정' if char_pred == 1 else '부정'}")
    print(f"TF-IDF 모델 예측: {'긍정' if tfidf_pred == 1 else '부정'}") 
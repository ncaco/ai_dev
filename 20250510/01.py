import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import FastICA, NMF
import soundfile as sf
import scipy.signal

def extract_mfcc(file_path, n_mfcc=20, max_pad_len=None):
    """
    음성 파일에서 MFCC 특징을 추출하는 함수
    
    Parameters:
        file_path (str): 음성 파일 경로
        n_mfcc (int): 추출할 MFCC 계수 개수
        max_pad_len (int): 패딩할 최대 길이 (None이면 패딩 없음)
    
    Returns:
        mfcc_features (np.ndarray): 추출된 MFCC 특징
    """
    # 음성 파일 로드
    y, sr = librosa.load(file_path, sr=22050)
    
    # MFCC 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 필요시 패딩 적용 (시계열 길이 통일)
    if max_pad_len is not None:
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
    return mfcc

def plot_mfcc(mfcc, title="MFCC"):
    """
    MFCC를 시각화하는 함수
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def mfcc_to_features(mfcc):
    """
    MFCC를 고정 길이 특징 벡터로 변환하는 함수 (통계 기반)
    """
    # 각 MFCC 계수에 대한 통계 특징 추출
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    max_val = np.max(mfcc, axis=1)
    min_val = np.min(mfcc, axis=1)
    
    # 모든 특징을 하나의 벡터로 결합
    features = np.concatenate([mean, std, max_val, min_val])
    return features

# 새로운 기능: 음성 분리 함수들

def separate_audio_ica(file_path, n_components=2, output_dir='separated_audio_ica'):
    """
    FastICA를 사용하여 음성 파일을 여러 독립 성분으로 분리하는 함수
    
    Parameters:
        file_path (str): 음성 파일 경로
        n_components (int): 분리할 성분 수
        output_dir (str): 분리된 음성 파일을 저장할 디렉토리
    
    Returns:
        separated_signals (list): 분리된 신호 리스트
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 오디오 로드
    y, sr = librosa.load(file_path, sr=None)
    
    # 스펙트로그램 계산 (STFT 사용)
    stft = librosa.stft(y)
    stft_mag, stft_phase = librosa.magphase(stft)
    
    # 스펙트로그램 크기를 로그 스케일로 변환 (ICA 입력용)
    log_spec = np.log1p(stft_mag)
    
    # 데이터 형태 변경: (주파수, 시간) -> (주파수, 시간) 행렬 평탄화
    X = log_spec.T  # 시간 x 주파수 형태로 변환
    
    # FastICA 적용
    ica = FastICA(n_components=n_components, random_state=42)
    S = ica.fit_transform(X)  # 분리된 신호의 표현
    A = ica.mixing_  # 혼합 행렬
    
    # 각 성분에 대한 원본 스펙트로그램 복원
    separated_specs = []
    for i in range(n_components):
        # i번째 성분만 선택
        S_i = np.zeros_like(S)
        S_i[:, i] = S[:, i]
        
        # 원래 공간으로 투영
        X_i = np.dot(S_i, A.T)
        
        # 원본 형태로 변환 (시간, 주파수) -> (주파수, 시간)
        log_spec_i = X_i.T
        
        # 로그 스케일에서 원래 스케일로 되돌림
        spec_i = np.expm1(log_spec_i)
        
        # 위상 정보와 결합하여 복소수 스펙트로그램 생성
        stft_complex_i = spec_i * stft_phase
        
        # 역 STFT를 통해 시간 도메인 신호로 변환
        y_i = librosa.istft(stft_complex_i)
        
        # 신호 정규화
        y_i = y_i / np.max(np.abs(y_i))
        
        # 결과 저장
        output_file = os.path.join(output_dir, f"component_{i+1}.wav")
        sf.write(output_file, y_i, sr)
        
        separated_specs.append(spec_i)
        
        # 스펙트로그램 시각화
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(spec_i, ref=np.max),
                                y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'분리된 성분 {i+1} 스펙트로그램')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"component_{i+1}_spec.png"))
        plt.close()
    
    print(f"{n_components}개 성분으로 음성이 분리되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    return [os.path.join(output_dir, f"component_{i+1}.wav") for i in range(n_components)]

def separate_audio_nmf(file_path, n_components=2, output_dir='separated_audio_nmf'):
    """
    NMF(Non-negative Matrix Factorization)를 사용하여 음성 파일을 여러 성분으로 분리하는 함수
    
    Parameters:
        file_path (str): 음성 파일 경로
        n_components (int): 분리할 성분 수
        output_dir (str): 분리된 음성 파일을 저장할 디렉토리
    
    Returns:
        separated_signals (list): 분리된 신호 파일 경로 리스트
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 오디오 로드
    y, sr = librosa.load(file_path, sr=None)
    
    # 스펙트로그램 계산
    S = np.abs(librosa.stft(y))
    
    # NMF 적용
    nmf = NMF(n_components=n_components, random_state=42)
    W = nmf.fit_transform(S.T)  # 시간-성분 행렬
    H = nmf.components_  # 성분-주파수 행렬
    
    # 원본 스펙트로그램의 위상 정보 저장
    phase = np.angle(librosa.stft(y))
    
    # 각 성분 분리 및 저장
    for i in range(n_components):
        # i번째 성분만 선택하여 재구성
        W_i = np.copy(W)
        W_i[:, [j for j in range(n_components) if j != i]] = 0
        S_i = np.dot(W_i, H).T
        
        # 위상 정보와 결합하여 복소수 스펙트로그램 생성
        S_complex_i = S_i * np.exp(1j * phase)
        
        # 역 STFT를 통해 시간 도메인 신호로 변환
        y_i = librosa.istft(S_complex_i)
        
        # 신호 정규화
        y_i = y_i / np.max(np.abs(y_i))
        
        # 결과 저장
        output_file = os.path.join(output_dir, f"component_{i+1}.wav")
        sf.write(output_file, y_i, sr)
        
        # 스펙트로그램 시각화
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(S_i, ref=np.max),
                                y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'NMF 분리 성분 {i+1} 스펙트로그램')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"component_{i+1}_spec.png"))
        plt.close()
    
    print(f"NMF를 사용하여 {n_components}개 성분으로 음성이 분리되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    return [os.path.join(output_dir, f"component_{i+1}.wav") for i in range(n_components)]

def separate_vocal_instrumental(file_path, output_dir='vocal_instrumental'):
    """
    음악에서 보컬(음성)과 백그라운드 음악을 분리하는 함수
    
    Parameters:
        file_path (str): 음악 파일 경로
        output_dir (str): 분리된 음성 파일을 저장할 디렉토리
    
    Returns:
        vocal_file (str): 보컬 파일 경로
        instrumental_file (str): 백그라운드 음악 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 오디오 로드
    y, sr = librosa.load(file_path, sr=None)
    
    # 스펙트로그램 계산
    D = librosa.stft(y)
    S, phase = librosa.magphase(D)
    
    # 중앙값 필터링을 사용하여 보컬과 백그라운드 분리
    # (보컬은 일반적으로 시간적으로 변동이 큼)
    S_filter = scipy.signal.medfilt2d(S, kernel_size=(1, 31))  # 주파수 방향으로는 필터링하지 않음
    
    # 보컬과 백그라운드 마스크 생성
    mask_vocal = S - S_filter
    mask_vocal = np.maximum(0, mask_vocal)  # 음수 값은 0으로 설정
    
    # 마스크 정규화
    mask_sum = mask_vocal + S_filter + 1e-10  # 0으로 나누는 것 방지
    mask_vocal = mask_vocal / mask_sum
    mask_inst = S_filter / mask_sum
    
    # 마스크 적용 및 신호 복원
    S_vocal = S * mask_vocal
    S_inst = S * mask_inst
    
    # 위상 정보를 사용하여 복소수 스펙트로그램 생성
    D_vocal = S_vocal * phase
    D_inst = S_inst * phase
    
    # 역 STFT를 통해 시간 도메인 신호로 변환
    y_vocal = librosa.istft(D_vocal)
    y_inst = librosa.istft(D_inst)
    
    # 신호 정규화
    y_vocal = y_vocal / np.max(np.abs(y_vocal))
    y_inst = y_inst / np.max(np.abs(y_inst))
    
    # 결과 저장
    vocal_file = os.path.join(output_dir, "vocal.wav")
    inst_file = os.path.join(output_dir, "instrumental.wav")
    
    sf.write(vocal_file, y_vocal, sr)
    sf.write(inst_file, y_inst, sr)
    
    # 스펙트로그램 시각화
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('원본 스펙트로그램')
    
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_vocal, ref=np.max),
                            y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('보컬 스펙트로그램')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_inst, ref=np.max),
                            y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('악기 스펙트로그램')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "separation_result.png"))
    plt.close()
    
    print(f"보컬과 악기 음원이 분리되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
    return vocal_file, inst_file

# 예제: 음성 파일 처리 및 시각화
def example_extraction():
    # 테스트용 음성 파일 (실제 경로로 변경 필요)
    audio_path = "test_audio.wav"
    
    # 음성 파일이 존재하는지 확인
    if not os.path.exists(audio_path):
        print(f"파일이 존재하지 않습니다: {audio_path}")
        print("이 예제를 실행하려면 음성 파일이 필요합니다.")
        return
    
    # MFCC 추출
    mfcc_features = extract_mfcc(audio_path)
    print("MFCC shape:", mfcc_features.shape)
    print(mfcc_features)
    
    # MFCC 시각화
    plot_mfcc(mfcc_features)
    
    # 머신러닝용 특징 추출
    ml_features = mfcc_to_features(mfcc_features)
    print("Machine Learning Feature Vector Shape:", ml_features.shape)

# 새로운 예제: 음성 분리 테스트
def example_audio_separation():
    # 테스트용 음성 파일 (실제 경로로 변경 필요)
    audio_path = "test_audio.wav"
    
    # 음성 파일이 존재하는지 확인
    if not os.path.exists(audio_path):
        print(f"파일이 존재하지 않습니다: {audio_path}")
        print("이 예제를 실행하려면 음성 파일이 필요합니다.")
        return
        
    print("\n1. FastICA를 사용한 음성 분리 (2개 성분)")
    separate_audio_ica(audio_path, n_components=2)
    
    print("\n2. NMF를 사용한 음성 분리 (2개 성분)")
    separate_audio_nmf(audio_path, n_components=2)
    
    print("\n3. 보컬/악기 분리 (음악 파일에 적합)")
    separate_vocal_instrumental(audio_path)

# 예제: 간단한 음성 분류 모델 (감정, 화자, 성별 등 분류)
def example_classification(data_folder="audio_data", categories=None):
    """
    음성 파일들을 분류하는 간단한 예제
    
    Parameters:
        data_folder (str): 음성 데이터가 있는 폴더
        categories (list): 분류할 카테고리 목록 (폴더 이름)
    """
    if categories is None:
        # 기본값 설정 (예: 감정 분류)
        categories = ["happy", "sad", "angry", "neutral"]
    
    # 폴더가 존재하는지 확인
    if not os.path.exists(data_folder):
        print(f"데이터 폴더가 존재하지 않습니다: {data_folder}")
        print("이 예제를 실행하려면 음성 데이터가 필요합니다.")
        return
    
    features = []
    labels = []
    
    # 각 카테고리 폴더에서 음성 파일 처리
    for i, category in enumerate(categories):
        category_folder = os.path.join(data_folder, category)
        
        if not os.path.exists(category_folder):
            print(f"카테고리 폴더가 존재하지 않습니다: {category_folder}")
            continue
        
        # 모든 오디오 파일 처리
        for file_name in os.listdir(category_folder):
            if file_name.endswith(".wav") or file_name.endswith(".mp3"):
                file_path = os.path.join(category_folder, file_name)
                
                # MFCC 추출
                mfcc = extract_mfcc(file_path)
                
                # 특징 벡터 생성
                feature_vector = mfcc_to_features(mfcc)
                
                # 데이터셋에 추가
                features.append(feature_vector)
                labels.append(i)  # 카테고리 인덱스를 레이블로 사용
    
    # numpy 배열로 변환
    X = np.array(features)
    y = np.array(labels)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 분류 모델 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 모델 평가
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=categories)
    
    print(f"모델 정확도: {accuracy:.4f}")
    print("분류 보고서:")
    print(report)
    
    return clf, scaler

# 메인 함수
def main():
    print("=== MFCC 추출 및 활용 예제 ===")
    print("\n1. 단일 음성 파일에서 MFCC 추출 및 시각화")
    example_extraction()
    
    print("\n2. 음성 분리 예제")
    print("(음성 파일을 여러 성분으로 분리합니다)")
    example_audio_separation()
    
    print("\n3. 음성 분류 모델 학습 예제")
    print("(이 예제는 감정/화자 등 카테고리별로 정리된 음성 데이터가 필요합니다)")
    # 실제 데이터가 있다면 아래 주석 해제
    # example_classification()
    
    print("\n추가 사용법:")
    print("- extract_mfcc(): 음성 파일에서 MFCC 추출")
    print("- plot_mfcc(): MFCC 시각화")
    print("- mfcc_to_features(): MFCC를 머신러닝 모델용 특징 벡터로 변환")
    print("- separate_audio_ica(): FastICA를 이용한 음성 분리")
    print("- separate_audio_nmf(): NMF를 이용한 음성 분리")
    print("- separate_vocal_instrumental(): 보컬과 악기 분리")

if __name__ == "__main__":
    main()

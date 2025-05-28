"""
오디오 처리를 위한 유틸리티 함수

이 모듈은 wav2vec 모델에 사용되는 오디오 파일 처리를 위한 다양한 유틸리티 함수를 제공합니다.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

def load_audio(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    오디오 파일을 로드하고 리샘플링합니다.
    
    Args:
        file_path: 오디오 파일 경로
        sample_rate: 목표 샘플링 레이트 (기본값: 16kHz)
        
    Returns:
        numpy.ndarray: 오디오 데이터
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        return audio
    except Exception as e:
        print(f"오디오 파일 로드 실패: {e}")
        return np.array([])

def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 16000) -> bool:
    """
    오디오 데이터를 파일로 저장합니다.
    
    Args:
        audio: 오디오 데이터
        file_path: 저장할 파일 경로
        sample_rate: 샘플링 레이트 (기본값: 16kHz)
        
    Returns:
        bool: 성공 여부
    """
    try:
        sf.write(file_path, audio, sample_rate)
        return True
    except Exception as e:
        print(f"오디오 파일 저장 실패: {e}")
        return False

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    오디오 데이터를 정규화합니다.
    
    Args:
        audio: 입력 오디오 데이터
        
    Returns:
        numpy.ndarray: 정규화된 오디오 데이터
    """
    if len(audio) == 0:
        return audio
        
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

def split_audio(audio: np.ndarray, 
                sample_rate: int = 16000, 
                segment_length: float = 10.0,
                overlap: float = 1.0) -> List[np.ndarray]:
    """
    긴 오디오 파일을 여러 세그먼트로 분할합니다.
    
    Args:
        audio: 입력 오디오 데이터
        sample_rate: 샘플링 레이트 (기본값: 16kHz)
        segment_length: 각 세그먼트의 길이(초) (기본값: 10초)
        overlap: 세그먼트 간 겹치는 시간(초) (기본값: 1초)
        
    Returns:
        List[numpy.ndarray]: 분할된 오디오 세그먼트 리스트
    """
    if len(audio) == 0:
        return []
        
    segment_samples = int(segment_length * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = segment_samples - overlap_samples
    
    # 오디오가 segment_samples보다 짧은 경우 패딩
    if len(audio) < segment_samples:
        padded_audio = np.zeros(segment_samples)
        padded_audio[:len(audio)] = audio
        return [padded_audio]
    
    segments = []
    for i in range(0, len(audio) - overlap_samples, step):
        end = i + segment_samples
        if end > len(audio):
            # 마지막 세그먼트가 완전하지 않은 경우
            segment = np.zeros(segment_samples)
            segment[:len(audio) - i] = audio[i:]
        else:
            segment = audio[i:end]
        segments.append(segment)
    
    return segments

def plot_waveform(audio: np.ndarray, 
                 sample_rate: int = 16000, 
                 title: str = "Audio Waveform",
                 save_path: Optional[str] = None) -> None:
    """
    오디오 파형을 시각화합니다.
    
    Args:
        audio: 입력 오디오 데이터
        sample_rate: 샘플링 레이트 (기본값: 16kHz)
        title: 그래프 제목 (기본값: "Audio Waveform")
        save_path: 저장할 이미지 경로 (기본값: None, 저장하지 않음)
    """
    if len(audio) == 0:
        print("시각화할 오디오 데이터가 없습니다.")
        return
        
    plt.figure(figsize=(12, 4))
    time = np.arange(0, len(audio)) / sample_rate
    plt.plot(time, audio)
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"파형 이미지 저장 완료: {save_path}")
    
    plt.show()

def plot_spectrogram(audio: np.ndarray, 
                    sample_rate: int = 16000,
                    title: str = "Spectrogram",
                    save_path: Optional[str] = None) -> None:
    """
    오디오 스펙트로그램을 시각화합니다.
    
    Args:
        audio: 입력 오디오 데이터
        sample_rate: 샘플링 레이트 (기본값: 16kHz)
        title: 그래프 제목 (기본값: "Spectrogram")
        save_path: 저장할 이미지 경로 (기본값: None, 저장하지 않음)
    """
    if len(audio) == 0:
        print("시각화할 오디오 데이터가 없습니다.")
        return
        
    plt.figure(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.subplot(211)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    plt.subplot(212)
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"스펙트로그램 이미지 저장 완료: {save_path}")
    
    plt.show()

def augment_audio(audio: np.ndarray, 
                 noise_level: float = 0.005,
                 shift_max: int = 1600,
                 speed_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    데이터 증강을 위해 오디오를 변형합니다.
    
    Args:
        audio: 입력 오디오 데이터
        noise_level: 추가할 노이즈의 강도 (기본값: 0.005)
        shift_max: 최대 시간 시프트 샘플 수 (기본값: 1600, 100ms @ 16kHz)
        speed_range: 속도 변화 범위 (기본값: 0.9 ~ 1.1)
        
    Returns:
        numpy.ndarray: 증강된 오디오 데이터
    """
    if len(audio) == 0:
        return audio
        
    augmented_audio = audio.copy()
    
    # 1. 노이즈 추가
    noise = np.random.randn(len(augmented_audio))
    augmented_audio += noise_level * noise
    
    # 2. 시간 시프트
    shift = np.random.randint(-shift_max, shift_max)
    if shift > 0:
        augmented_audio = np.pad(augmented_audio, (shift, 0), mode='constant')[:-shift]
    else:
        augmented_audio = np.pad(augmented_audio, (0, -shift), mode='constant')[-shift:]
    
    # 3. 속도 변화
    speed_factor = np.random.uniform(*speed_range)
    augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=speed_factor)
    
    # 원본 길이에 맞게 조정
    if len(augmented_audio) > len(audio):
        augmented_audio = augmented_audio[:len(audio)]
    elif len(augmented_audio) < len(audio):
        augmented_audio = np.pad(augmented_audio, (0, len(audio) - len(augmented_audio)), mode='constant')
    
    return augmented_audio

def get_audio_length(file_path: str) -> float:
    """
    오디오 파일의 길이(초)를 반환합니다.
    
    Args:
        file_path: 오디오 파일 경로
        
    Returns:
        float: 오디오 길이(초)
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return len(audio) / sr
    except Exception as e:
        print(f"오디오 길이 계산 실패: {e}")
        return 0.0

def batch_convert_audio_format(input_dir: str, 
                              output_dir: str, 
                              target_format: str = "wav",
                              sample_rate: int = 16000) -> None:
    """
    디렉토리 내의 모든 오디오 파일을 지정된 형식으로 변환합니다.
    
    Args:
        input_dir: 입력 오디오 파일 디렉토리
        output_dir: 출력 오디오 파일 디렉토리
        target_format: 목표 오디오 형식 (기본값: "wav")
        sample_rate: 목표 샘플링 레이트 (기본값: 16kHz)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            try:
                input_path = os.path.join(input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + f".{target_format}"
                output_path = os.path.join(output_dir, output_filename)
                
                audio = load_audio(input_path, sample_rate=sample_rate)
                save_audio(audio, output_path, sample_rate=sample_rate)
                
                print(f"변환 완료: {input_path} -> {output_path}")
            except Exception as e:
                print(f"파일 변환 실패 {filename}: {e}")

if __name__ == "__main__":
    # 사용 예시
    print("오디오 유틸리티 모듈입니다. 다른 스크립트에서 import하여 사용하세요.") 
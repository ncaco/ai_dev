%% 1) Load Ionosphere for Binary Classification
clear; clc; close all;

data = readtable('D3-KAGGLE-BC-Ionosphere.csv');
X = data(:, 1:end-1);
class = data(:, end);

% 여기까지 수행하면 분류학습기를 수행했을 때
% 목표 변수를 설정할 수 없다.
% 명령 창에 다음을 입력해 보자.
% >> y
'y'은(는) 인식할 수 없는 함수 또는 변수입니다.

%% 1-1) Text Label to Integer
X = table2array(X);
class = table2array(class);
% 사실, 여기까지만 실행시켜도 분류 학습기를 수행하는 데에는 문제가 없다.

% 'class'가 범주형 데이터가 아니라고 가정하고, 범주형으로 변환
classCategorical = categorical(class);
% grp2idx를 사용하여 각 클래스에 대한 고유한 숫자 식별자 할당
y = grp2idx(classCategorical) - 1; 
%% 1) Load Ionosphere for Binary Classification
clear; clc; close all;

data = readtable('D3-KAGGLE-BC-Ionoshpere.csv');
X = data(:, 1:end-1);
class = data(:, end);

% 여기까지 수행하면 분류학습기를 수행했을 때
%   목표 변수를 설정할 수 없다.
% 명령 창에 다음을 입력해 보자.
% >> y

%% 1-1) Text Label to Integer
X = table2array(X);
class = table2array(class);
% 사실, 여기까지만 실행시켜도 분류 학습기를 수행하는 데에는
%   문제가 없다.

% 'class'가 범주형 데이터가 아니라고 가정하고, 범주형으로 변환
classCategorical = categorical(class);
% grp2idx를 사용하여 각 클래스에 대한 고유한 숫자 식별자 할당
y = grp2idx(classCategorical) - 1;

%% 1-2) Train a Decision Tree Model
% 데이터를 훈련 세트와 테스트 세트로 나눕니다.
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);

% 훈련 데이터와 테스트 데이터로 분리
XTrain = X(idxTrain, :);
yTrain = y(idxTrain);
XTest = X(idxTest, :);
yTest = y(idxTest);

% 의사결정 트리 모델 훈련
treeModel = fitctree(XTrain, yTrain);

%% 1-3) Model Prediction and Performance Evaluation
% 테스트 데이터에 대한 예측 수행
[yPred, score] = predict(treeModel, XTest);

% 성능 평가
% 여기서는 단순히 정확도를 계산합니다.
accuracy = sum(yPred == yTest) / length(yTest);
fprintf('Accuracy of the decision tree model: %.2f%%\n', accuracy * 100);

% score 출력 (score는 각 클래스에 속할 확률을 포함하는 행렬)
disp('Scores for the test set:');
disp(score);

%% 1-4)
view(treeModel, 'Mode', 'graph');

%% 1-5)
% 점수에 기반한 ROC 플롯 생성
% 'XCrit'와 'YCrit'는 ROC 곡선의 X축과 Y축에 해당하는 비율
[XCrit, YCrit, T, AUC] = perfcurve(yTest, score(:,2), 1);

% ROC 플롯 그리기
figure(1);
plot(XCrit, YCrit, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend(sprintf('AUC = %.3f', AUC));

% 임계값(T)과 AUC 출력
disp(table(T, 'VariableNames', {'Thresholds'}));
fprintf('Area Under Curve (AUC): %.3f\n', AUC);

grid on;
grid minor;


%% 1-5) Scoring and Ranking
% 테스트 데이터에 대한 예측 점수 얻기
[~, scores] = predict(treeModel, XTest);

% 점수를 기반으로 테스트 데이터의 인덱스 순위 매기기
[~, rankIndex] = sort(scores(:, 2), 'descend');

% 실제 테스트 레이블 인덱스 추출하기
testIndices = find(idxTest);

% 순위에 따라 정렬된 레이블 추출
sortedLabels = class(testIndices(rankIndex));

% 테스트 데이터의 순위 생성
ranking = (1:length(rankIndex))';

% 순위에 따라 정렬된 점수 추출
sortedScores = scores(rankIndex, 2);

% 결과를 테이블로 생성
resultsTable = table(ranking, sortedLabels, sortedScores, 'VariableNames', {'Rank', 'Species', 'Score'});

% 결과 출력
disp(resultsTable);
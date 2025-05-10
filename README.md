## Kaggle 코드 분석과 연습

0. 심층 강화 학습을 활용한 주식 거래
https://www.kaggle.com/code/stpeteishii/nvidia-stock-trading-decision-sac/notebook

1. 필수 패키지 
gymnasium, numpy, pandas, matplotlib, stable-baselines3 

2. 다우존스 30 지수 데이터 다운로드 (CSV로 저장)

3. CSV 파일에서 데이터 로드, 전처리

4. 데이터에 기술 지표 추가
    - 기술 지표를 추가하여 데이터를 개선 
    - 기술 지표(예: 이동 평균, RSI, MACD)
    - 지표를 계산하고 데이터에 추가하는 방법
  
5. 사용자 지정 거래 환경(StockTradingEnv) 생성
    - 관찰 공간과 행동 공간
      (거래 비용 시뮬레이션 방법에 대한 정보를 포함)
      (환경 설정에 사용된 특정 속성이나 방법 등 설정)

6. 학습 모델 선언 - 앙상블 + PPO + A2C + DDPG + SAC + TD3

7. 테스트 모델 - 앙상블 + PPO + A2C + DDPG
    - 에이전트의 성능을 테스트
    - 평가 지표(예: 수익률, 변동성, 샤프 지수)
    - 시각화

    (DRL을 사용한 캐글의 코드를 분석해보았다)

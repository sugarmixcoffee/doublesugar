"""
강화학습을 위한 트레이딩 환경 모듈
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import gym
from gym import spaces
import gc  # 명시적 가비지 컬렉션 사용

from src.config.config import (
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    LOGGER
)

class TradingEnvironment:
    """
    강화학습을 위한 트레이딩 환경 클래스
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT,
        symbol: str = None
    ):
        """
        TradingEnvironment 클래스 초기화
        
        Args:
            data: 학습에 사용할 주식 데이터 (정규화된 데이터)
            window_size: 관측 윈도우 크기
            initial_balance: 초기 자본금
            max_trading_units: 최대 거래 단위
            transaction_fee_percent: 거래 수수료 비율
            symbol: 주식 심볼 (로깅용)
        """
        # 데이터프레임의 날짜 인덱스 처리 - 날짜 컬럼이 존재하면 제거
        if isinstance(data, pd.DataFrame):
            # 인덱스가 날짜 타입인지 확인
            if isinstance(data.index, pd.DatetimeIndex):
                # 인덱스를 리셋하여 숫자 인덱스로 변경
                self.data = data.reset_index(drop=True)
            else:
                self.data = data
        else:
            self.data = data
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_trading_units = max_trading_units
        self.transaction_fee_percent = transaction_fee_percent
        self.symbol = symbol if symbol else "UNKNOWN"
        
        # 데이터 관련 변수
        self.feature_dim = self.data.shape[1]
        self.data_length = len(self.data)
        
        # 환경 상태 변수
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission = 0
        
        # 이전 상태 추적을 위한 변수
        self._previous_commission = 0
        self._previous_shares_held = 0
        self._previous_portfolio_value = initial_balance
        self._no_trade_steps = 0
        
        # 에피소드 히스토리
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        # 행동 공간: [-1.0, 1.0] 범위의 연속적인 값
        # -1.0은 최대 매도, 0.0은 홀드, 1.0은 최대 매수
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 관측 공간: 가격 데이터 + 포트폴리오 상태
        # 가격 데이터: window_size x feature_dim
        # 포트폴리오 상태: [보유 현금 비율, 보유 주식 비율]
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=0, high=1, shape=(self.window_size, self.feature_dim), dtype=np.float32
            ),
            'portfolio_state': spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            )
        })
        
        LOGGER.info(f"{self.symbol} 트레이딩 환경 초기화 완료: 데이터 길이 {self.data_length}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        환경 초기화
        
        Returns:
            초기 관측값
        """
        # 상태 초기화
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission = 0
        
        # 이전 상태 변수 초기화
        self._previous_commission = 0
        self._previous_shares_held = 0
        self._previous_portfolio_value = self.initial_balance
        self._no_trade_steps = 0
        
        # 히스토리 초기화
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        
        # 초기 관측값 반환
        return self._get_observation()
    
    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행
        
        Args:
            action: 에이전트의 행동 (-1.0 ~ 1.0 범위의 값)
            
        Returns:
            (관측값, 보상, 종료 여부, 추가 정보) 튜플
        """
        # 행동 기록
        self.actions_history.append(action)
        
        # 이전 포트폴리오 가치 계산
        prev_portfolio_value = self._get_portfolio_value()
        
        # 행동 실행
        self._execute_trade_action(action)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 현재 포트폴리오 가치 계산
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_values_history.append(current_portfolio_value)
        
        # 보상 계산
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        self.rewards_history.append(reward)
        
        # 종료 여부 확인
        done = self.current_step >= self.data_length - 1
        
        # 관측값 및 추가 정보 반환
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        현재 관측값 반환 - 메모리 최적화 버전
        
        Returns:
            관측값 딕셔너리
        """
        # 윈도우 크기만큼의 가격 데이터 가져오기
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # 데이터가 충분하지 않은 경우 패딩 처리 (메모리 효율적 방식)
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            padding_size = self.window_size - (end_idx - start_idx)
            
            # view 또는 참조를 사용하여 불필요한 복사 방지
            actual_data = self.data.iloc[start_idx:end_idx].values
            
            # 패딩 포함한 배열 직접 생성 (메모리 효율적인 할당)
            market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            market_data[-len(actual_data):] = actual_data
        else:
            # 충분한 데이터가 있는 경우 (최적화된 방식)
            if end_idx - start_idx == self.window_size:
                # 정확히 윈도우 크기만큼 데이터가 있는 경우
                # 메모리 효율성: 복사 대신 view 사용
                market_data = self.data.iloc[start_idx:end_idx].values.astype(np.float32)
            else:
                # 데이터 길이가 윈도우 크기보다 작은 경우 효율적인 패딩
                actual_data = self.data.iloc[start_idx:end_idx].values
                market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
                market_data[-len(actual_data):] = actual_data
        
        # 포트폴리오 상태 계산 (불필요한 중간 연산 제거)
        current_price = self._get_current_price()
        shares_value = self.shares_held * current_price
        portfolio_value = self.balance + shares_value
        
        # 0 나누기 방지
        if portfolio_value <= 0:
            portfolio_state = np.array([1.0, 0.0], dtype=np.float32)  # 기본값: 현금 100%
        else:
            cash_ratio = self.balance / portfolio_value
            stock_ratio = shares_value / portfolio_value
            portfolio_state = np.array([cash_ratio, stock_ratio], dtype=np.float32)
        
        # 관측값 딕셔너리 생성 - 깊은 복사 없이 뷰 사용
        observation = {
            'market_data': market_data,  # 이미 astype으로 형변환되어 있음
            'portfolio_state': portfolio_state
        }
        
        # 상태 기록 (선택적으로 비활성화 가능)
        # 메모리 사용량이 중요한 경우 아래 줄을 주석 처리하고
        # 학습 로직에서 상태 히스토리를 필요로 하지 않는지 확인하세요
        self.states_history.append(observation)
        
        # 명시적 가비지 컬렉션을 통한 메모리 효율성 개선
        # 매 100 스텝마다 가비지 컬렉션 수행
        if self.current_step % 100 == 0:
            gc.collect()
        
        return observation
    
    def _execute_trade_action(self, action: float) -> None:
        """
        거래 행동 실행
        
        Args:
            action: 에이전트의 행동 (-1.0 ~ 1.0 범위의 값)
        """
        current_price = self._get_current_price()
        
        if current_price <= 0:
            LOGGER.warning(f"현재 가격이 0 이하입니다: {current_price}")
            return
        
        # 행동 값을 거래 단위로 변환
        action_value = action[0] if isinstance(action, np.ndarray) else action
        # print(f"[DEBUG] step {self.current_step}: action_value = {action_value}")
        
        # 거래 임계값 설정 (0.1)
        TRADE_THRESHOLD = 0.1
        
        if abs(action_value) < TRADE_THRESHOLD:
            # print(f"[DEBUG] step {self.current_step}: action_value {action_value:.4f} is below threshold {TRADE_THRESHOLD}, no trade executed")
            return
            
        if action_value > 0:  # 매수
            # 매수할 수 있는 최대 주식 수 계산
            max_affordable = self.balance / (current_price * (1 + self.transaction_fee_percent))
            # 행동 값에 따라 매수할 주식 수 결정 (0 ~ max_trading_units 범위)
            shares_to_buy = min(
                max_affordable,
                self.max_trading_units * (action_value / TRADE_THRESHOLD)  # 임계값으로 정규화
            )
            
            # 정수로 반올림
            shares_to_buy = int(shares_to_buy)
            # print(f"[DEBUG] step {self.current_step}: computed shares_to_buy = {shares_to_buy}")
            
            if shares_to_buy > 0:
                # 매수 비용 계산
                buy_cost = shares_to_buy * current_price
                # 수수료 계산
                commission = buy_cost * self.transaction_fee_percent
                # 총 비용
                total_cost = buy_cost + commission
                
                # 잔고가 충분한지 확인
                if self.balance >= total_cost:
                    # 매수 실행
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
                    self.total_shares_purchased += shares_to_buy
                    self.total_commission += commission
                    
                    # 평균 매수 단가 업데이트
                    if self.shares_held > 0:
                        self.cost_basis = ((self.cost_basis * (self.shares_held - shares_to_buy)) + buy_cost) / self.shares_held
                    
                    LOGGER.debug(f"매수: {shares_to_buy}주 @ {current_price:.2f}, 비용: {total_cost:.2f}, 수수료: {commission:.2f}")
        
        elif action_value < 0:  # 매도
            # 매도할 주식 수 결정 (0 ~ shares_held 범위)
            shares_to_sell = min(
                self.shares_held,
                self.max_trading_units * (abs(action_value) / TRADE_THRESHOLD)  # 임계값으로 정규화
            )
            
            # 정수로 반올림
            shares_to_sell = int(shares_to_sell)
            # print(f"[DEBUG] step {self.current_step}: computed shares_to_sell = {shares_to_sell}")
            
            if shares_to_sell > 0:
                # 매도 수익 계산
                sell_value = shares_to_sell * current_price
                # 수수료 계산
                commission = sell_value * self.transaction_fee_percent
                # 순 수익
                net_value = sell_value - commission
                
                # 매도 실행
                self.balance += net_value
                self.shares_held -= shares_to_sell
                self.total_shares_sold += shares_to_sell
                self.total_sales_value += sell_value
                self.total_commission += commission
                
                LOGGER.debug(f"매도: {shares_to_sell}주 @ {current_price:.2f}, 수익: {net_value:.2f}, 수수료: {commission:.2f}")
    
    def _get_current_price(self) -> float:
        """
        현재 주가 반환
        
        Returns:
            현재 종가
        """
        # 'close' 열의 인덱스 찾기 (일반적으로 4번째 열)
        close_idx = 3
        
        # 현재 스텝의 종가 가져오기
        normalized_price = self.data.iloc[self.current_step, close_idx]
        
        # 정규화된 가격이 0 이하인 경우 처리
        if normalized_price <= 0:
            if self.current_step > 0:
                # 이전 스텝의 가격 사용
                normalized_price = self.data.iloc[self.current_step - 1, close_idx]
            else:
                # 첫 스텝인 경우 기본값 사용
                normalized_price = 0.5  # 정규화된 범위(0~1)의 중간값
        
        # 정규화된 가격을 실제 가격 범위로 변환 (예: $100 ~ $200 범위)
        BASE_PRICE = 150.0  # AAPL의 평균적인 가격대
        PRICE_RANGE = 50.0  # 가격 변동 범위
        actual_price = BASE_PRICE + (normalized_price - 0.5) * PRICE_RANGE
        
        return max(actual_price, 0.01)  # 최소 가격 보장
    
    def _get_portfolio_value(self) -> float:
        """
        현재 포트폴리오 가치 계산
        
        Returns:
            포트폴리오 총 가치
        """
        return self.balance + self.shares_held * self._get_current_price()
    
    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        """
        개선된 보상 계산 함수
        
        Args:
            prev_portfolio_value: 이전 포트폴리오 가치
            current_portfolio_value: 현재 포트폴리오 가치
            
        Returns:
            보상값
        """
        if prev_portfolio_value <= 0:
            return 0
            
        # 수익률 계산
        return_rate = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # 현재 스텝의 수수료만 고려
        current_commission = self.total_commission - self._previous_commission
        commission_penalty = current_commission / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # 포지션 변화 계산
        position_change = abs(self.shares_held - self._previous_shares_held)
        position_change_ratio = position_change / self.max_trading_units if self.max_trading_units > 0 else 0
        
        # 기본 보상 (수익률)
        reward = return_rate * 100  # 스케일링 (1% 변화 = 1.0 보상)
        
        # 수수료 페널티 (가중치 부여)
        reward -= commission_penalty * 200
        
        # 홀딩 보너스: 작은 포지션 변화 또는 홀딩 시 보너스
        if position_change_ratio < 0.1:  # 포지션 변화가 작은 경우
            self._no_trade_steps += 1
            holding_bonus = min(0.005 * self._no_trade_steps, 0.05)  # 최대 0.05까지 보너스 (이전 0.1에서 감소)
            reward += holding_bonus
        else:
            self._no_trade_steps = 0
            
        # 과도한 거래 페널티
        if position_change_ratio > 0.5:  # 큰 포지션 변화에 대한 페널티
            trade_penalty = position_change_ratio * 0.5
            reward -= trade_penalty
            
        # 시간 기반 패널티 수정 (선형)
        if self._no_trade_steps >= 60:  # 60스텝 이상 거래가 없는 경우
            # 지수 함수를 사용하여 패널티 계산 (0.001부터 시작하여 최대 -0.4까지)
            steps_over_60 = self._no_trade_steps - 60
            time_penalty = -min (0.001 * (steps_over_60+1), 0.4)  # 최대 0.2까지 패널티
            # time_penalty = max(time_penalty, -0.4)  # 최대 -0.4로 제한
            reward += time_penalty
            
        # 보상 클리핑 (-10 ~ 10 범위로 제한)
        reward = np.clip(reward, -10.0, 10.0)
        
        # 이전 상태 업데이트
        self._previous_commission = self.total_commission
        self._previous_shares_held = self.shares_held
        self._previous_portfolio_value = current_portfolio_value
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """
        추가 정보 반환
        
        Returns:
            추가 정보 딕셔너리
        """
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        # 수익률 계산
        if self.initial_balance > 0:
            total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        else:
            total_return = 0
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'cost_basis': self.cost_basis,
            'total_shares_purchased': self.total_shares_purchased,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'total_commission': self.total_commission
        }
    
    def render(self, mode: str = 'human') -> None:
        """
        환경 시각화
        
        Args:
            mode: 시각화 모드
        """
        info = self._get_info()
        
        print(f"Step: {info['step']}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Shares held: {info['shares_held']}")
        print(f"Current price: ${info['current_price']:.2f}")
        print(f"Portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Total return: {info['total_return'] * 100:.2f}%")
        print(f"Total commission paid: ${info['total_commission']:.2f}")
        print("-" * 50)
    
    def get_episode_data(self) -> Dict[str, List]:
        """
        에피소드 데이터 반환
        
        Returns:
            에피소드 데이터 딕셔너리
        """
        return {
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'portfolio_values': self.portfolio_values_history
        }
    
    def get_final_portfolio_value(self) -> float:
        """
        최종 포트폴리오 가치 반환
        
        Returns:
            최종 포트폴리오 가치
        """
        return self._get_portfolio_value()
    
    def get_total_reward(self) -> float:
        """
        총 보상 반환
        
        Returns:
            에피소드의 총 보상
        """
        return sum(self.rewards_history)

    def set_data(self, data: pd.DataFrame) -> None:
        """
        환경의 데이터를 설정 (한 번만 호출)
        
        Args:
            data: 정규화된 시장 데이터
        """
        self.data = data
        self.feature_dim = len(data.columns)
        self.current_step = self.window_size
        self.max_steps = len(data) - self.window_size
        LOGGER.info(f"환경 데이터 설정 완료: {self.feature_dim}개 특성, {self.max_steps}개 스텝")


class MultiAssetTradingEnvironment:
    """
    다중 자산 트레이딩 환경 클래스
    """
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    ):
        """
        MultiAssetTradingEnvironment 클래스 초기화
        
        Args:
            data_dict: 심볼을 키로 하고 정규화된 데이터를 값으로 하는 딕셔너리
            window_size: 관측 윈도우 크기
            initial_balance: 초기 자본금
            max_trading_units: 최대 거래 단위
            transaction_fee_percent: 거래 수수료 비율
        """
        # 전처리된 데이터 딕셔너리 생성 (날짜 인덱스 처리)
        processed_data_dict = {}
        for symbol, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                # 인덱스가 날짜 타입인지 확인
                if isinstance(data.index, pd.DatetimeIndex):
                    # 인덱스를 리셋하여 숫자 인덱스로 변경
                    processed_data_dict[symbol] = data.reset_index(drop=True)
                else:
                    processed_data_dict[symbol] = data
            else:
                processed_data_dict[symbol] = data
        
        # 각 자산에 대해 개별 환경 생성
        self.envs = {}
        self.symbols = list(processed_data_dict.keys())
        
        for symbol, data in processed_data_dict.items():
            self.envs[symbol] = TradingEnvironment(
                data=data,
                window_size=window_size,
                initial_balance=initial_balance / len(processed_data_dict),  # 자산별 균등 배분
                max_trading_units=max_trading_units,
                transaction_fee_percent=transaction_fee_percent,
                symbol=symbol
            )
        
        # 환경 관련 변수
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_trading_units = max_trading_units
        self.transaction_fee_percent = transaction_fee_percent
        
        # 행동 공간: 각 자산에 대해 [-1.0, 1.0] 범위의 연속적인 값
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.symbols),), dtype=np.float32
        )
        
        # 관측 공간: 각 자산의 관측 공간을 딕셔너리로 구성
        self.observation_space = spaces.Dict({
            symbol: env.observation_space for symbol, env in self.envs.items()
        })
        
        LOGGER.info(f"다중 자산 트레이딩 환경 초기화 완료: {len(self.symbols)}개 자산")
    
    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        환경 초기화
        
        Returns:
            초기 관측값
        """
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
        
        return observations
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행
        
        Args:
            actions: 심볼을 키로 하고 행동을 값으로 하는 딕셔너리
            
        Returns:
            (관측값, 보상, 종료 여부, 추가 정보) 튜플
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # 각 자산에 대한 행동 실행
        for symbol, env in self.envs.items():
            action = actions.get(symbol, 0.0)  # 행동이 없는 경우 홀드
            obs, rew, done, info = env.step(action)
            
            observations[symbol] = obs
            rewards[symbol] = rew
            dones[symbol] = done
            infos[symbol] = info
        
        # 전체 보상은 각 자산의 보상 평균
        total_reward = sum(rewards.values()) / len(self.symbols)
        
        # 모든 자산의 에피소드가 종료되면 전체 에피소드 종료
        done = all(dones.values())
        
        # 전체 포트폴리오 가치 계산
        total_portfolio_value = sum(info['portfolio_value'] for info in infos.values())
        
        # 추가 정보에 전체 포트폴리오 가치 포함
        infos['total'] = {
            'portfolio_value': total_portfolio_value,
            'total_return': (total_portfolio_value - self.initial_balance) / self.initial_balance
        }
        
        return observations, total_reward, done, infos
    
    def render(self, mode: str = 'human') -> None:
        """
        환경 시각화
        
        Args:
            mode: 시각화 모드
        """
        total_portfolio_value = 0
        
        print("=" * 50)
        print("다중 자산 트레이딩 환경 상태")
        print("=" * 50)
        
        for symbol, env in self.envs.items():
            info = env._get_info()
            total_portfolio_value += info['portfolio_value']
            
            print(f"자산: {symbol}")
            print(f"  가격: ${info['current_price']:.2f}")
            print(f"  보유량: {info['shares_held']}")
            print(f"  포트폴리오 가치: ${info['portfolio_value']:.2f}")
            print(f"  수익률: {info['total_return'] * 100:.2f}%")
            print("-" * 50)
        
        total_return = (total_portfolio_value - self.initial_balance) / self.initial_balance
        print(f"총 포트폴리오 가치: ${total_portfolio_value:.2f}")
        print(f"총 수익률: {total_return * 100:.2f}%")
        print("=" * 50)
    
    def get_final_portfolio_value(self) -> float:
        """
        최종 포트폴리오 가치 반환
        
        Returns:
            최종 포트폴리오 가치
        """
        return sum(env.get_final_portfolio_value() for env in self.envs.values())
    
    def get_total_reward(self) -> float:
        """
        총 보상 반환
        
        Returns:
            에피소드의 총 보상
        """
        return sum(env.get_total_reward() for env in self.envs.values()) / len(self.symbols)


if __name__ == "__main__":
    # 모듈 테스트 코드
    import matplotlib.pyplot as plt
    from src.data_collection.data_collector import DataCollector
    from src.preprocessing.data_processor import DataProcessor
    
    # 데이터 수집 및 전처리
    collector = DataCollector(symbols=["AAPL"])
    data = collector.load_all_data()
    
    if not data:
        print("저장된 데이터가 없어 데이터를 수집합니다.")
        data = collector.collect_and_save()
    
    if data:
        # 데이터 전처리
        processor = DataProcessor()
        results = processor.process_all_symbols(data)
        
        # 환경 생성 및 테스트
        if "AAPL" in results:
            # 정규화된 데이터 사용
            normalized_data = results["AAPL"]["normalized_data"]
            
            # 환경 생성
            env = TradingEnvironment(data=normalized_data, symbol="AAPL")
            
            # 환경 테스트
            obs = env.reset()
            done = False
            total_reward = 0
            
            # 랜덤 행동으로 테스트
            while not done:
                action = np.random.uniform(-1.0, 1.0)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if env.current_step % 100 == 0:
                    env.render()
            
            # 최종 결과 출력
            print("\n최종 결과:")
            print(f"총 보상: {total_reward:.2f}")
            print(f"최종 포트폴리오 가치: ${env.get_final_portfolio_value():.2f}")
            print(f"총 수익률: {(env.get_final_portfolio_value() - env.initial_balance) / env.initial_balance * 100:.2f}%")
            
            # 포트폴리오 가치 변화 시각화
            episode_data = env.get_episode_data()
            plt.figure(figsize=(12, 6))
            plt.plot(episode_data['portfolio_values'])
            plt.title('포트폴리오 가치 변화')
            plt.xlabel('스텝')
            plt.ylabel('포트폴리오 가치 ($)')
            plt.grid(True, alpha=0.3)
            plt.savefig('./results/portfolio_value_test.png')
            plt.close() 
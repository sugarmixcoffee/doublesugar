"""
SAC 모델 학습을 위한 트레이너 모듈
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm
import gc  # 명시적 가비지 컬렉션
import psutil  # 메모리 사용량 모니터링

from src.config.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    MODELS_DIR,
    RESULTS_DIR,
    LOGGER,
    MEMORY_CLEANUP_INTERVAL,
    LOW_MEMORY_MODE,
    USE_GPU_MEMORY,
    MAX_GPU_MEMORY_USAGE,
    GPU_BATCH_SIZE
)
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.utils.utils import create_directory, plot_learning_curve, plot_equity_curve, get_timestamp

class Trainer:
    """
    SAC 모델 학습을 위한 트레이너 클래스
    """
    
    def __init__(
        self,
        agent: SACAgent,
        env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
        batch_size: int = BATCH_SIZE,
        num_episodes: int = NUM_EPISODES,
        evaluate_interval: int = EVALUATE_INTERVAL,
        save_interval: int = SAVE_MODEL_INTERVAL,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        models_dir: Union[str, Path] = MODELS_DIR,
        results_dir: Union[str, Path] = RESULTS_DIR
    ):
        """
        Trainer 클래스 초기화
        
        Args:
            agent: 학습할 SAC 에이전트
            env: 학습에 사용할 트레이딩 환경
            batch_size: 배치 크기
            num_episodes: 학습할 총 에피소드 수
            evaluate_interval: 평가 간격 (에피소드 단위)
            save_interval: 모델 저장 간격 (에피소드 단위)
            max_steps: 에피소드당 최대 스텝 수
            models_dir: 모델 저장 디렉토리
            results_dir: 결과 저장 디렉토리
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # 디렉토리 생성
        create_directory(self.models_dir)
        create_directory(self.results_dir)
        
        # 학습 통계
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.train_losses = []
        
        LOGGER.info(f"Trainer 초기화 완료: {num_episodes}개 에피소드, 배치 크기 {batch_size}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        SAC 모델 학습 수행 - GPU 메모리(VRAM) 최적화 버전
        
        Returns:
            학습 통계 딕셔너리
        """
        start_time = time.time()
        timestamp = get_timestamp()
        
        LOGGER.info(f"학습 시작: {self.num_episodes}개 에피소드")
        
        # GPU 메모리 사용 여부 확인
        use_gpu = USE_GPU_MEMORY and torch.cuda.is_available()
        
        # 시스템 자원 모니터링 초기화
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB 단위
        
        # 리소스 사용량 로깅
        if use_gpu:
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB 단위
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB 단위
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위
            LOGGER.info(f"초기 리소스 사용량: CPU 메모리 {initial_memory:.2f} MB, GPU 메모리 {gpu_allocated:.2f} GB / {gpu_total:.2f} GB (예약됨: {gpu_reserved:.2f} GB)")
            
            # CUDA 스트림 생성 (병렬 처리를 위해)
            train_stream = torch.cuda.Stream()
        else:
            LOGGER.info(f"초기 리소스 사용량: CPU 메모리 {initial_memory:.2f} MB, GPU 사용 안함")
        
        # 학습 배치 크기 최적화
        effective_batch_size = min(self.batch_size, GPU_BATCH_SIZE) if use_gpu else self.batch_size
        if effective_batch_size != self.batch_size:
            LOGGER.info(f"GPU 메모리 최적화를 위해 배치 크기 조정: {self.batch_size} -> {effective_batch_size}")
        
        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            episode_loss = {
                "actor_loss": 0,
                "critic_loss": 0,
                "alpha_loss": 0,
                "entropy": 0,
                "alpha": 0
            }
            episode_steps = 0
            done = False
            # 에피소드별 거래 통계 추가
            action_count = 0  # 실제 거래 횟수
            profit_trades = 0  # 수익 거래 횟수
            loss_trades = 0  # 손실 거래 횟수
            initial_portfolio_value = self.env._get_portfolio_value()  # 초기 포트폴리오 가치
            previous_shares = 0  # 이전 보유 주식 수
            # 승률 측정 (추가한 부분)
            previous_portfolio_value = initial_portfolio_value # 이전 포폴 가치
            total_trades = 0    # 전체 거래 횟수
            winning_trades = 0  # 승리한 거래 횟수
            
            # 에피소드 진행
            while not done and episode_steps < self.max_steps:
                # GPU 메모리 사용 시 비동기 처리 (CUDA 스트림 활용)
                if use_gpu:
                    with torch.cuda.stream(train_stream):
                        # 행동 선택 (비동기 GPU 계산)
                        action = self.agent.select_action(state)
                else:
                    # 일반 CPU 계산
                    action = self.agent.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = self.env.step(action)
                
                # 거래 여부 확인 (주식 보유량 변화로 판단)
                current_shares = info.get('shares_held', 0)
                current_portfolio_value = info.get('portfolio_value', previous_portfolio_value)
                
                if current_shares != previous_shares:
                    action_count += 1
                    total_trades += 1
                    # 포트폴리오 가치가 증가했는지 확인
                    if current_portfolio_value > previous_portfolio_value:
                        winning_trades += 1
                    previous_portfolio_value = current_portfolio_value
                previous_shares = current_shares
                
                # # 거래 여부 확인 (주식 보유량 변화로 판단)
                # current_shares = info.get('shares_held', 0)
                # if current_shares != previous_shares:
                #     action_count += 1
                #     # 매도 거래인 경우 수익/손실 판단
                #     if current_shares < previous_shares:
                #         # 매도 거래 - 수익/손실 계산은 reward로 판단
                #         if reward > 0:
                #             profit_trades += 1
                #         else:
                #             loss_trades += 1
                # previous_shares = current_shares
                
                # 경험 저장
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # 모델 업데이트
                if len(self.agent.replay_buffer) > effective_batch_size:
                    # GPU 메모리 사용 시 비동기 처리
                    if use_gpu:
                        with torch.cuda.stream(train_stream):
                            loss = self.agent.update_parameters(effective_batch_size)
                    else:
                        loss = self.agent.update_parameters(effective_batch_size)
                    
                    # 손실 누적
                    for k, v in loss.items():
                        episode_loss[k] += v
                
                # 다음 상태로 전환 (참조 복사 방지)
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # GPU 메모리 최적화 - 주기적인 동기화 및 메모리 정리
                if use_gpu and episode_steps % 100 == 0:
                    # CUDA 스트림 동기화
                    torch.cuda.synchronize()
                    
                    # 현재 GPU 메모리 사용량 확인
                    gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # 메모리 사용량이 임계치를 초과하는 경우 정리
                    if gpu_allocated > gpu_total * MAX_GPU_MEMORY_USAGE:
                        LOGGER.warning(f"GPU 메모리 사용량이 높습니다: {gpu_allocated:.2f} GB / {gpu_total:.2f} GB")
                        torch.cuda.empty_cache()
            
            # 참조 해제 및 메모리 정리
            del state, next_state, action, reward, done, info
            
            # 에피소드 통계 기록
            if episode_steps > 0:
                # 에피소드 평균 보상 계산
                episode_reward = episode_reward / episode_steps
                
                # 손실 평균 계산
                for k in episode_loss:
                    episode_loss[k] /= episode_steps
            
            # 최종 포트폴리오 가치와 승률 계산
            final_portfolio_value = self.env._get_portfolio_value()
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            
            # # 최종 포트폴리오 가치와 승률 계산
            # final_portfolio_value = self.env._get_portfolio_value()
            # total_trades = profit_trades + loss_trades
            # win_rate = (profit_trades / total_trades * 100) if total_trades > 0 else 0.0

           # 수익률 계산
            return_rate = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100) if initial_portfolio_value > 0 else 0.0
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.train_losses.append(episode_loss)
            
            # 진행 상황 로깅 (CPU/GPU 메모리 모니터링 포함)
            episode_time = time.time() - episode_start_time
            current_cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB 단위
            
            # GPU 메모리 사용량 로깅
            if use_gpu:
                gpu_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB 단위
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB 단위
                LOGGER.info(f"에피소드 {episode}/{self.num_episodes} - 평균 보상: {episode_reward:.4f}, 스텝: {episode_steps}, "
                           f"시간: {episode_time:.2f}초, CPU: {current_cpu_memory:.2f} MB, GPU: {gpu_allocated:.2f} GB (예약됨: {gpu_reserved:.2f} GB), "
                           f"액션수: {action_count}, 포트폴리오: ${final_portfolio_value:.2f}, 수익률: {return_rate:.2f}%, 승률: {win_rate:.1f}%")
            else:
                LOGGER.info(f"에피소드 {episode}/{self.num_episodes} - 평균 보상: {episode_reward:.4f}, 스텝: {episode_steps}, "
                           f"시간: {episode_time:.2f}초, CPU: {current_cpu_memory:.2f} MB, "
                           f"액션수: {action_count}, 포트폴리오: ${final_portfolio_value:.2f}, 수익률: {return_rate:.2f}%, 승률: {win_rate:.1f}%")
            # 주기적 평가
            if episode % self.evaluate_interval == 0:
                # GPU 동기화 및 메모리 정리
                if use_gpu:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # CPU 메모리 정리
                gc.collect()
                
                # 평가 수행
                eval_reward = self.evaluate(max_steps=self.max_steps)
                self.eval_rewards.append(eval_reward)
                
                # 평가 결과 로깅
                if use_gpu:
                    gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                    LOGGER.info(f"평가 결과 (에피소드 {episode}) - 보상: {eval_reward:.2f}, GPU: {gpu_allocated:.2f} GB")
                else:
                    LOGGER.info(f"평가 결과 (에피소드 {episode}) - 보상: {eval_reward:.2f}")
            
            # 주기적 모델 저장
            if episode % self.save_interval == 0:
                # 저장 전 메모리 정리
                if use_gpu:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                model_path = self.agent.save_model(self.models_dir, f"episode_{episode}_")
                LOGGER.info(f"모델 저장 완료: {model_path}")
            
            # 학습 곡선 업데이트
            if episode % 10 == 0:
                self._plot_training_curves(timestamp)
            
            # 주기적 메모리 정리
            if episode % MEMORY_CLEANUP_INTERVAL == 0:
                before_cpu_cleanup = process.memory_info().rss / (1024 * 1024)
                
                # CPU 메모리 정리
                gc.collect()
                
                # GPU 메모리 정리
                if use_gpu:
                    before_gpu_cleanup = torch.cuda.memory_allocated() / (1024**3)
                    torch.cuda.empty_cache()
                    after_gpu_cleanup = torch.cuda.memory_allocated() / (1024**3)
                    gpu_saved = before_gpu_cleanup - after_gpu_cleanup
                
                after_cpu_cleanup = process.memory_info().rss / (1024 * 1024)
                cpu_saved = before_cpu_cleanup - after_cpu_cleanup
                
                if use_gpu:
                    LOGGER.info(f"메모리 정리: CPU {before_cpu_cleanup:.2f} MB -> {after_cpu_cleanup:.2f} MB (절약: {cpu_saved:.2f} MB), "
                               f"GPU {before_gpu_cleanup:.2f} GB -> {after_gpu_cleanup:.2f} GB (절약: {gpu_saved:.2f} GB)")
                else:
                    LOGGER.info(f"메모리 정리: CPU {before_cpu_cleanup:.2f} MB -> {after_cpu_cleanup:.2f} MB (절약: {cpu_saved:.2f} MB)")
        
        # 최종 모델 저장
        # 저장 전 메모리 정리
        if use_gpu:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        final_model_path = self.agent.save_model(self.models_dir, "final_")
        LOGGER.info(f"최종 모델 저장 완료: {final_model_path}")
        
        # 최종 학습 곡선 저장
        self._plot_training_curves(timestamp)
        
        # 학습 시간 계산
        total_time = time.time() - start_time
        final_cpu_memory = process.memory_info().rss / (1024 * 1024)
        
        # 최종 리소스 사용량 로깅
        if use_gpu:
            final_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            LOGGER.info(f"학습 완료: 총 시간 {total_time:.2f}초 ({total_time/60:.2f}분), "
                       f"CPU 메모리: {final_cpu_memory:.2f} MB, GPU 메모리: {final_gpu_memory:.2f} GB")
        else:
            LOGGER.info(f"학습 완료: 총 시간 {total_time:.2f}초 ({total_time/60:.2f}분), CPU 메모리: {final_cpu_memory:.2f} MB")
        
        # 최종 메모리 정리
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
        
        # 학습 통계 반환
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "actor_losses": [loss["actor_loss"] for loss in self.train_losses],
            "critic_losses": [loss["critic_loss"] for loss in self.train_losses],
            "alpha_losses": [loss["alpha_loss"] for loss in self.train_losses],
            "entropy_values": [loss["entropy"] for loss in self.train_losses]
        }
    
    def evaluate(self, num_episodes: int = 1, max_steps: int = MAX_STEPS_PER_EPISODE) -> float:
        """
        현재 정책 평가 - GPU 메모리 최적화 버전
        
        Args:
            num_episodes: 평가할 에피소드 수
            max_steps: 에피소드당 최대 스텝 수
            
        Returns:
            평균 에피소드 보상
        """
        from src.config.config import MAX_EVAL_TIME, USE_GPU_MEMORY
        
        total_reward = 0
        start_time = time.time()
        max_eval_time = MAX_EVAL_TIME  # 최대 평가 시간 (초)
        
        # GPU 메모리 사용 여부 확인
        use_gpu = USE_GPU_MEMORY and torch.cuda.is_available()
        

        
        # 평가용 CUDA 스트림 생성 (학습과 분리)
        if use_gpu:
            eval_stream = torch.cuda.Stream()
            # 평가 전 GPU 메모리 정리
            torch.cuda.empty_cache()
            # 현재 GPU 메모리 상태 확인
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB 단위
            LOGGER.debug(f"평가 시작 GPU 메모리: {gpu_allocated:.2f} GB")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            # 에피소드 진행 (스텝 제한 및 시간 제한 추가)
            while not done and steps < max_steps:
                # 시간 제한 체크
                if time.time() - start_time > max_eval_time:
                    LOGGER.warning(f"평가 시간 제한 초과: {max_eval_time}초, 평가 종료")
                    break
                
                # GPU 메모리 사용 시 비동기 처리
                if use_gpu:
                    with torch.cuda.stream(eval_stream):
                        # 평가 모드에서 행동 선택 (비동기 GPU 계산)
                        action = self.agent.select_action(state, evaluate=True)
                    # 주기적으로 스트림 동기화
                    if steps % 10 == 0:
                        torch.cuda.synchronize()
                else:
                    # 일반 CPU 계산
                    action = self.agent.select_action(state, evaluate=True)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # GPU 메모리 모니터링 및 필요시 정리
                if use_gpu and steps % 50 == 0:
                    gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # 메모리 사용량이 90% 이상인 경우 정리
                    if gpu_allocated > gpu_total * 0.9:
                        LOGGER.warning(f"평가 중 GPU 메모리 사용량이 높습니다: {gpu_allocated:.2f} GB / {gpu_total:.2f} GB")
                        # 스트림 동기화 후 메모리 정리
                        torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
            # 평균 보상 계산
            if steps > 0:
                episode_reward = episode_reward / steps
                
            total_reward += episode_reward
            
            # 시간 제한 체크
            if time.time() - start_time > max_eval_time:
                LOGGER.warning(f"평가 시간 제한 초과: {max_eval_time}초, 남은 에피소드 평가 건너뜀")
                # 평가한 에피소드 수로 평균 계산
                return total_reward / (episode + 1)
            
            # GPU 메모리 정리 (각 에피소드 후)
            if use_gpu:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        eval_time = time.time() - start_time
        
        # 평가 결과 로깅 (GPU 메모리 정보 포함)
        if use_gpu:
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            LOGGER.info(f"평가 완료: {num_episodes}개 에피소드, 소요 시간: {eval_time:.2f}초, GPU 메모리: {gpu_allocated:.2f} GB")
        else:
            LOGGER.info(f"평가 완료: {num_episodes}개 에피소드, 소요 시간: {eval_time:.2f}초")
        
        return total_reward / num_episodes
    
    def _plot_training_curves(self, timestamp: str) -> None:
        """학습 곡선 플로팅 및 저장

        Args:
            timestamp: 타임스탬프 (파일명용)
        """
        # 결과 디렉토리 생성
        result_dir = os.path.join("results", f"training_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # 에피소드 보상 플로팅
        plt.figure(figsize=(10, 5))
        # x_points = list(range(self.evaluate_interval, len(self.episode_rewards) * self.evaluate_interval + 1, self.evaluate_interval))
        x_points = list(range(1, len(self.episode_rewards) + 1))
        plt.plot(x_points, self.episode_rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        reward_plot_path = os.path.join(result_dir, "episode_rewards.png")
        plt.savefig(reward_plot_path)
        plt.close()
        
        LOGGER.info(f"학습 곡선 저장됨: {os.path.abspath(reward_plot_path)}")

        # 손실값 추출
        actor_losses = [loss["actor_loss"] for loss in self.train_losses]
        critic_losses = [loss["critic_loss"] for loss in self.train_losses]
        alpha_losses = [loss["alpha_loss"] for loss in self.train_losses]

        # 손실 플로팅
        plt.figure(figsize=(15, 5))
        x_points = list(range(1, len(self.train_losses) + 1))
        
        plt.subplot(1, 3, 1)
        plt.plot(x_points, actor_losses, label='Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(x_points, critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(x_points, alpha_losses, label='Alpha Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Alpha Loss')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 손실 그래프 저장
        loss_plot_path = os.path.join(result_dir, "training_losses.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        LOGGER.info(f"손실 곡선 저장됨: {os.path.abspath(loss_plot_path)}")


if __name__ == "__main__":
    # 모듈 테스트 코드
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
        
        if "AAPL" in results:
            # 정규화된 데이터 사용
            normalized_data = results["AAPL"]["normalized_data"]
            
            # 환경 생성
            env = TradingEnvironment(data=normalized_data, symbol="AAPL")
            
            # 에이전트 생성
            state_dim = env.observation_space['market_data'].shape[0] * env.observation_space['market_data'].shape[1] + env.observation_space['portfolio_state'].shape[0]
            action_dim = 1
            
            # CNN 모델 사용
            agent = SACAgent(
                action_dim=action_dim,
                input_shape=(env.window_size, env.feature_dim),
                use_cnn=True
            )
            
            # 트레이너 생성 및 학습
            trainer = Trainer(
                agent=agent,
                env=env,
                num_episodes=100,  # 테스트용으로 적은 에피소드
                batch_size=64,
                evaluate_interval=10,
                save_interval=50
            )
            
            # 학습 실행
            training_stats = trainer.train()
            
            print(f"학습 완료: 최종 평가 보상 {trainer.eval_rewards[-1] if trainer.eval_rewards else 'N/A'}") 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment
from src.utils.logger import Logger
from src.config.config import Config


class Backtester:
    """
    백테스팅 모듈: 학습된 SAC 에이전트를 사용하여 과거 데이터에서 성능을 테스트합니다.
    """
    
    def __init__(
        self,
        agent: SACAgent,
        test_data: pd.DataFrame,
        config: Config,
        logger: Optional[Logger] = None,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0025,
        benchmark_data: Optional[pd.Series] = None,
    ):
        """
        Backtester 클래스 초기화
        
        Args:
            agent: 학습된 SAC 에이전트
            test_data: 테스트할 과거 데이터
            config: 설정 객체
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
            initial_balance: 초기 자본금
            commission_rate: 거래 수수료율 (0.0025 = 0.25%)
            benchmark_data: 벤치마크 데이터 (예: S&P 500 지수)
        """
        self.agent = agent
        self.test_data = test_data.copy()
        self.config = config
        self.logger = logger
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.benchmark_data = benchmark_data
        
        # 데이터 유효성 검사
        self._validate_data()
        
        # 백테스트 환경 설정
        self.env = TradingEnvironment(
            data=self.test_data,
            window_size=config.window_size,
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            reward_scaling=getattr(config, 'reward_scaling', 1.0),
            reward_function=getattr(config, 'reward_function', 'profit'),
            max_shares=getattr(config, 'max_shares', None),
            allow_short=getattr(config, 'allow_short', False),
            render_mode=None,
        )
        
        # 결과 저장 변수
        self.results = {
            "portfolio_values": [],
            "returns": [],
            "actions": [],
            "positions": [],
            "rewards": [],
            "timestamps": [],
            "trades": [],
            "benchmark_values": [],
        }
        
    def _validate_data(self) -> None:
        """데이터 유효성 검사"""
        if self.test_data.empty:
            raise ValueError("Test data cannot be empty")
        
        if len(self.test_data) < self.config.window_size:
            raise ValueError(f"Test data length ({len(self.test_data)}) must be greater than window size ({self.config.window_size})")
        
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # NaN 값 확인
        if self.test_data[required_columns].isnull().any().any():
            warnings.warn("Test data contains NaN values. Consider preprocessing the data.")
        
    def run_backtest(self, verbose: bool = True) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            verbose: 진행 상황을 표시할지 여부
        
        Returns:
            백테스트 결과를 담은 딕셔너리
        """
        try:
            # 환경 초기화
            state, info = self.env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            # 벤치마크 초기화 (매수 후 보유 전략)
            if self.benchmark_data is not None:
                benchmark_initial_price = self.benchmark_data.iloc[self.config.window_size]
                benchmark_shares = self.initial_balance / benchmark_initial_price
            
            # 백테스트 진행 상황 표시 설정
            max_steps = len(self.test_data) - self.config.window_size - 1
            iterator = tqdm(range(max_steps), desc="Backtesting") if verbose else range(max_steps)
            
            # 에피소드 진행
            for step in iterator:
                if done:
                    break
                    
                # 에이전트로부터 행동 선택
                with torch.no_grad():
                    action = self.agent.select_action(state, evaluate=True)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # 결과 저장
                self.results["portfolio_values"].append(info.get("portfolio_value", self.initial_balance))
                self.results["returns"].append(info.get("return", 0.0))
                self.results["actions"].append(action)
                self.results["positions"].append(info.get("position", 0))
                self.results["rewards"].append(reward)
                self.results["timestamps"].append(info.get("timestamp", step))
                
                # 벤치마크 가치 계산
                if self.benchmark_data is not None and step + self.config.window_size < len(self.benchmark_data):
                    current_benchmark_price = self.benchmark_data.iloc[step + self.config.window_size]
                    benchmark_value = benchmark_shares * current_benchmark_price
                    self.results["benchmark_values"].append(benchmark_value)
                
                # 거래 기록 저장
                if info.get("trade_executed", False):
                    trade_info = {
                        "timestamp": info.get("timestamp", step),
                        "action": action,
                        "price": info.get("current_price", 0.0),
                        "shares": info.get("trade_shares", 0),
                        "cost": info.get("trade_cost", 0.0),
                        "position": info.get("position", 0),
                        "portfolio_value": info.get("portfolio_value", self.initial_balance),
                        "commission": info.get("commission", 0.0)
                    }
                    self.results["trades"].append(trade_info)
                
                # 상태 및 보상 업데이트
                state = next_state
                total_reward += reward
                step_count += 1
                
                # 진행 상황 업데이트
                if verbose and step % 100 == 0:
                    current_portfolio_value = info.get("portfolio_value", self.initial_balance)
                    iterator.set_postfix({
                        'Portfolio': f'${current_portfolio_value:.2f}',
                        'Return': f'{((current_portfolio_value/self.initial_balance-1)*100):.2f}%'
                    })
                
            # 결과가 비어있지 않은지 확인
            if not self.results["portfolio_values"]:
                raise RuntimeError("No data was collected during backtest")
                
            # 배열로 변환
            self._convert_results_to_arrays()
            
            # 성능 지표 계산
            self.calculate_metrics()
            
            # 로깅
            if self.logger:
                final_value = self.results['portfolio_values'][-1]
                self.logger.info(f"Backtest completed with final portfolio value: ${final_value:.2f}")
                self.logger.info(f"Total return: {((final_value/self.initial_balance-1)*100):.2f}%")
                self.logger.info(f"Total reward: {total_reward:.2f}")
                self.logger.info(f"Total steps: {step_count}")
                if 'metrics' in self.results:
                    self.logger.info(f"Sharpe ratio: {self.results['metrics'].get('sharpe_ratio', 'N/A'):.3f}")
            
            return self.results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during backtesting: {str(e)}")
            raise
    
    def _convert_results_to_arrays(self) -> None:
        """결과 리스트를 numpy 배열로 변환"""
        for key in ["portfolio_values", "returns", "actions", "positions", "rewards"]:
            if key in self.results and self.results[key]:
                self.results[key] = np.array(self.results[key])
            else:
                self.results[key] = np.array([])
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        백테스트 결과에서 성능 지표 계산
        
        Returns:
            성능 지표를 담은 딕셔너리
        """
        if len(self.results["portfolio_values"]) == 0:
            return {}
            
        # 일별 수익률 계산
        portfolio_values = self.results["portfolio_values"]
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        daily_returns = pd.Series(daily_returns).fillna(0)
        
        # 누적 수익률
        final_value = portfolio_values[-1]
        cumulative_return = (final_value / self.initial_balance) - 1
        
        # 연간화된 수익률 (252 트레이딩 데이)
        n_days = len(daily_returns)
        if n_days > 0:
            annual_return = ((1 + cumulative_return) ** (252 / n_days)) - 1
        else:
            annual_return = 0
        
        # 변동성 계산
        daily_std = daily_returns.std()
        annual_volatility = daily_std * np.sqrt(252) if not np.isnan(daily_std) else 0
        
        # 샤프 비율 (무위험 이자율 0% 가정)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 최대 낙폭 (Maximum Drawdown)
        portfolio_series = pd.Series(portfolio_values)
        cumulative_max = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # 승률 계산 개선
        win_rate = self._calculate_win_rate()
        
        # 평균 거래 수익
        avg_trade_return = self._calculate_avg_trade_return()
        
        # 변동성 조정 수익률 (Calmar Ratio)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 벤치마크 대비 성과
        benchmark_metrics = self._calculate_benchmark_metrics() if self.benchmark_data is not None else {}
        
        # 지표 저장
        metrics = {
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "total_trades": len(self.results["trades"]),
            "avg_daily_return": daily_returns.mean(),
            "return_skewness": daily_returns.skew(),
            "return_kurtosis": daily_returns.kurtosis(),
            **benchmark_metrics
        }
        
        self.results["metrics"] = metrics
        return metrics
    
    def _calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.results["trades"]:
            return 0.0
            
        profitable_trades = 0
        for i, trade in enumerate(self.results["trades"]):
            if i == 0:
                continue  # 첫 번째 거래는 비교할 이전 가격이 없음
                
            prev_trade = self.results["trades"][i-1]
            if trade["portfolio_value"] > prev_trade["portfolio_value"]:
                profitable_trades += 1
                
        return profitable_trades / max(len(self.results["trades"]) - 1, 1)
    
    def _calculate_avg_trade_return(self) -> float:
        """평균 거래 수익률 계산"""
        if len(self.results["trades"]) < 2:
            return 0.0
            
        trade_returns = []
        for i in range(1, len(self.results["trades"])):
            prev_value = self.results["trades"][i-1]["portfolio_value"]
            curr_value = self.results["trades"][i]["portfolio_value"]
            trade_return = (curr_value - prev_value) / prev_value
            trade_returns.append(trade_return)
            
        return np.mean(trade_returns) if trade_returns else 0.0
    
    def _calculate_benchmark_metrics(self) -> Dict[str, float]:
        """벤치마크 대비 성과 지표 계산"""
        if not self.benchmark_data or not self.results["benchmark_values"]:
            return {}
            
        benchmark_values = np.array(self.results["benchmark_values"])
        portfolio_values = self.results["portfolio_values"][:len(benchmark_values)]
        
        if len(benchmark_values) == 0 or len(portfolio_values) == 0:
            return {}
            
        # 벤치마크 수익률
        benchmark_return = (benchmark_values[-1] / benchmark_values[0]) - 1
        
        # 알파 계산 (포트폴리오 수익률 - 벤치마크 수익률)
        portfolio_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        alpha = portfolio_return - benchmark_return
        
        return {
            "benchmark_return": benchmark_return,
            "alpha": alpha,
            "relative_performance": alpha
        }
    
    def save_results(self, filepath: str) -> None:
        """
        백테스트 결과를 JSON 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        try:
            # 결과 저장을 위한 딕셔너리 생성
            save_data = {
                "backtest_info": {
                    "initial_balance": self.initial_balance,
                    "commission_rate": self.commission_rate,
                    "test_period": {
                        "start": str(self.test_data.index[0]) if hasattr(self.test_data.index[0], '__str__') else "N/A",
                        "end": str(self.test_data.index[-1]) if hasattr(self.test_data.index[-1], '__str__') else "N/A",
                        "total_days": len(self.test_data)
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "performance": {
                    "final_portfolio_value": float(self.results["portfolio_values"][-1]) if len(self.results["portfolio_values"]) > 0 else self.initial_balance,
                    "total_trades": len(self.results["trades"]),
                    "metrics": self.results.get("metrics", {})
                },
                "trades": self.results["trades"][:100],  # 최근 100개 거래만 저장
                "config": {
                    "window_size": self.config.window_size,
                    "model_type": "SAC",
                }
            }
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False, default=str)
                
            if self.logger:
                self.logger.info(f"Backtest results saved to {filepath}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def plot_portfolio_performance(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        포트폴리오 성능 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
            figsize: 그래프 크기
        """
        if len(self.results["portfolio_values"]) == 0:
            print("No data to plot")
            return
            
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Portfolio Performance Analysis", fontsize=16, fontweight='bold')
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. 포트폴리오 가치 변화
        ax1 = axes[0, 0]
        portfolio_values = self.results["portfolio_values"]
        ax1.plot(portfolio_values, label="Portfolio Value", color="royalblue", linewidth=2)
        
        # 벤치마크 비교
        if self.results["benchmark_values"]:
            ax1.plot(self.results["benchmark_values"][:len(portfolio_values)], 
                    label="Benchmark", color="orange", linewidth=2, alpha=0.7)
        
        # 거래 지점 표시 (처음 50개만)
        trades_to_show = self.results["trades"][:50]
        for trade in trades_to_show:
            if hasattr(trade["timestamp"], '__index__'):
                idx = trade["timestamp"]
                if idx < len(portfolio_values):
                    if trade["action"] > 0:  # 매수
                        ax1.scatter(idx, portfolio_values[idx], color="green", marker="^", s=50, alpha=0.7)
                    elif trade["action"] < 0:  # 매도
                        ax1.scatter(idx, portfolio_values[idx], color="red", marker="v", s=50, alpha=0.7)
        
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 일별 수익률
        ax2 = axes[0, 1]
        if len(portfolio_values) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            ax2.plot(daily_returns, color="green", alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title("Daily Returns")
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Daily Return")
            ax2.grid(True, alpha=0.3)
        
        # 3. 포지션 변화
        ax3 = axes[1, 0]
        if len(self.results["positions"]) > 0:
            ax3.plot(self.results["positions"], color="purple", linewidth=2)
            ax3.set_title("Position Over Time")
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Position")
            ax3.grid(True, alpha=0.3)
        
        # 4. 성능 지표 테이블
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if 'metrics' in self.results:
            metrics = self.results['metrics']
            metrics_text = f"""
            Performance Metrics:
            
            Total Return: {metrics.get('cumulative_return', 0):.2%}
            Annual Return: {metrics.get('annual_return', 0):.2%}
            Volatility: {metrics.get('annual_volatility', 0):.2%}
            Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
            Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
            Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
            Win Rate: {metrics.get('win_rate', 0):.2%}
            Total Trades: {metrics.get('total_trades', 0)}
            """
            
            if 'alpha' in metrics:
                metrics_text += f"\nAlpha: {metrics['alpha']:.2%}"
                
            ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Portfolio performance plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        """
        낙폭(Drawdown) 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        if len(self.results["portfolio_values"]) == 0:
            print("No data to plot")
            return
            
        plt.figure(figsize=(14, 7))
        
        # 포트폴리오 가치
        portfolio_values = pd.Series(self.results["portfolio_values"])
        
        # 누적 최대값
        cumulative_max = portfolio_values.cummax()
        
        # 낙폭 계산
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        # 낙폭 플롯
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='crimson', alpha=0.3, label='Drawdown')
        plt.plot(drawdown, color='crimson', linestyle='-', linewidth=1)
        
        # 최대 낙폭 표시
        if len(drawdown) > 0:
            max_dd = drawdown.min()
            max_dd_idx = drawdown.argmin()
            plt.scatter(max_dd_idx, max_dd, color='darkred', marker='o', s=100, 
                       label=f'Max Drawdown: {max_dd:.2%}')
        
        # 그래프 레이블 및 제목
        plt.title("Portfolio Drawdown Analysis", fontsize=16)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Drawdown (%)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_returns_distribution(self, save_path: Optional[str] = None) -> None:
        """
        수익률 분포 시각화
        
        Args:
            save_path: 그래프를 저장할 파일 경로 (옵션)
        """
        if len(self.results["portfolio_values"]) < 2:
            print("Insufficient data to plot returns distribution")
            return
            
        plt.figure(figsize=(14, 7))
        
        # 일별 수익률
        portfolio_values = self.results["portfolio_values"]
        daily_returns = pd.Series(np.diff(portfolio_values) / portfolio_values[:-1])
        daily_returns = daily_returns.dropna()
        
        if len(daily_returns) == 0:
            print("No valid returns data to plot")
            return
        
        # 수익률 분포 플롯
        plt.hist(daily_returns, bins=50, density=True, alpha=0.7, color="royalblue", edgecolor='black')
        
        # KDE 플롯 추가
        try:
            from scipy import stats
            x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
            kde = stats.gaussian_kde(daily_returns)
            plt.plot(x, kde(x), color='red', linewidth=2, label='KDE')
        except ImportError:
            pass
        
        # 0선 표시
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Zero Return')
        
        # 평균 수익률 표시
        mean_return = daily_returns.mean()
        plt.axvline(x=mean_return, color='green', linestyle='-', alpha=0.7, 
                   label=f'Mean Return: {mean_return:.4%}')
        
        # 통계 정보
        stats_text = f"""
        Statistics:
        Mean: {mean_return:.4%}
        Std Dev: {daily_returns.std():.4%}
        Min: {daily_returns.min():.4%}
        Max: {daily_returns.max():.4%}
        Skewness: {daily_returns.skew():.3f}
        Kurtosis: {daily_returns.kurtosis():.3f}
        """
        
        plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # 그래프 레이블 및 제목
        plt.title("Distribution of Daily Returns", fontsize=16)
        plt.xlabel("Daily Return", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.logger:
                self.logger.info(f"Returns distribution plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self) -> str:
        """백테스트 결과 리포트 생성"""
        if 'metrics' not in self.results:
            return "No metrics available. Please run backtest first."
            
        metrics = self.results['metrics']
        
        report = f"""
        ==========================================
        BACKTESTING REPORT
        ==========================================
        
        Test Period: {len(self.results['portfolio_values'])} days
        Initial Balance: ${self.initial_balance:,.2f}
        Final Portfolio Value: ${self.results['portfolio_values'][-1]:,.2f}
        
        PERFORMANCE METRICS:
        ------------------------------------------
        Total Return: {metrics.get('cumulative_return', 0):.2%}
        Annualized Return: {metrics.get('annual_return', 0):.2%}
        Annualized Volatility: {metrics.get('annual_volatility', 0):.2%}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
        Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
        Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}
        
        TRADING STATISTICS:
        ------------------------------------------
        Total Trades: {metrics.get('total_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.2%}
        Average Trade Return: {metrics.get('avg_trade_return', 0):.4%}
        
        RISK METRICS:
        ------------------------------------------
        Return Skewness: {metrics.get('return_skewness', 0):.3f}
        Return Kurtosis: {metrics.get('return_kurtosis', 0):.3f}
        Average Daily Return: {metrics.get('avg_daily_return', 0):.4%}
        """
        
        if 'alpha' in metrics:
            report += f"""
        BENCHMARK COMPARISON:
        ------------------------------------------
        Benchmark Return: {metrics.get('benchmark_return', 0):.2%}
        Alpha (Excess Return): {metrics.get('alpha', 0):.2%}
        """
        
        report += "\n=========================================="
        
        return report
            
    def visualize_results(self, save_dir: Optional[str] = None) -> None:
        """
        백테스트 결과를 시각화하고 저장
        
        Args:
            save_dir: 시각화 결과를 저장할 디렉토리 경로 (옵션)
        """
        # 저장 디렉토리 설정
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 포트폴리오 성능 차트
            portfolio_path = os.path.join(save_dir, f"portfolio_performance_{timestamp}.png")
            self.plot_portfolio_performance(portfolio_path)
            
            # 낙폭 차트
            drawdown_path = os.path.join(save_dir, f"drawdown_{timestamp}.png")
            self.plot_drawdown(drawdown_path)
            
            # 수익률 분포 차트
            returns_path = os.path.join(save_dir, f"returns_distribution_{timestamp}.png")
            self.plot_returns_distribution(returns_path)
            
            # 결과 데이터 저장
            results_path = os.path.join(save_dir, f"backtest_results_{timestamp}.json")
            self.save_results(results_path)
        else:
            # 시각화만 수행
            self.plot_portfolio_performance()
            self.plot_drawdown()
            self.plot_returns_distribution() 
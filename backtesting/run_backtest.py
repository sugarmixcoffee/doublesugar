#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Dict
from pathlib import Path

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment
from src.utils.logger import Logger
from src.config import config
from src.backtesting.backtester import Backtester
from src.backtesting.visualizer import Visualizer
from src.utils.utils import create_directory, get_timestamp, load_from_csv

from src.config.config import (
    DEVICE,
    TARGET_SYMBOLS,
    TRANSACTION_FEE_PERCENT,
    INITIAL_BALANCE,
    LOGGER,
    DATA_DIR,
    ALPHA_INIT 
)
def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 모델 백테스트 실행')
    
    # 필수 인자
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델의 경로')
    parser.add_argument('--config_path', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--data_path', type=str, required=True,
                        help='테스트 데이터 경로')
    
    # 선택적 인자 (기본값 설정)
    parser.add_argument('--results_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='초기 자본금')
    parser.add_argument('--transaction_fee_percent', type=float, default=0.001,
                        help='거래 수수료율 (0.0025 = 0.25%)')
    parser.add_argument('--benchmark_data_path', type=str, default=None,
                        help='벤치마크 데이터 경로 (옵션)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='관측 창 크기 (설정에서 가져오지 않을 경우)')
    parser.add_argument('--symbols', nargs='+', help='대상 심볼 목록')
    parser.add_argument('--output_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    
    # SAC 에이전트 파라미터 - 명령행에서 직접 지정 가능
    parser.add_argument('--state_dim', type=int, default=None,
                        help='상태 공간의 차원 (직접 지정 시 최우선 적용)')
    parser.add_argument('--action_dim', type=int, default=1,     
                        help='행동 공간의 차원')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='은닉층의 차원 (직접 지정 시 최우선 적용)')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='Actor 학습률')
    parser.add_argument('--critic_lr', type=float, default=1e-4,
                        help='Critic 학습률')
    parser.add_argument('--alpha_lr', type=float, default=1e-4,
                        help='Alpha 학습률')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='할인율')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='타겟 네트워크 업데이트 비율')
    parser.add_argument('--alpha_init', type=float, default=0.1,
                        help='초기 엔트로피 계수')
    parser.add_argument('--target_update_interval', type=int, default=2,
                        help='타겟 네트워크 업데이트 간격')
    parser.add_argument('--use_automatic_entropy_tuning', type=bool, default=True,
                        help='자동 엔트로피 조정 사용 여부')
    parser.add_argument('--buffer_capacity', type=int, default=500000,
                        help='리플레이 버퍼 크기')
    parser.add_argument('--use_cnn', type=bool, default=False,
                        help='CNN 사용 여부')
    parser.add_argument('--use_gpu_memory', type=bool, default=True,
                        help='GPU 메모리 사용 여부')
    
    return parser.parse_args()


def calculate_state_dim(env, use_cnn: bool = False) -> int:
    """환경에서 동적으로 상태 차원 계산"""
    try:
        sample_obs = env.reset()
        
        if isinstance(sample_obs, dict):
            # Dictionary observation space
            market_dim = np.prod(sample_obs['market_data'].shape)
            portfolio_dim = np.prod(sample_obs['portfolio_state'].shape)
            total_dim = market_dim + portfolio_dim
            LOGGER.info(f"상태 차원 계산: market={market_dim}, portfolio={portfolio_dim}, total={total_dim}")
            return total_dim
        else:
            # Array observation space
            total_dim = np.prod(sample_obs.shape)
            LOGGER.info(f"상태 차원 계산: {total_dim}")
            return total_dim
            
    except Exception as e:
        LOGGER.warning(f"상태 차원 자동 계산 실패: {e}")
        # 환경 타입에 따른 기본값 추정
        if hasattr(env, 'feature_dim') and hasattr(env, 'window_size'):
            estimated_dim = env.feature_dim * env.window_size + 3  # +3 for portfolio state
            LOGGER.info(f"상태 차원 추정: {estimated_dim}")
            return estimated_dim
        else:
            LOGGER.error("상태 차원을 계산할 수 없습니다")
            return None


def infer_model_params_from_checkpoint(model_path: str) -> dict:
    """체크포인트에서 모델 파라미터 추론 (config는 무시)"""
    try:
        # 경로 확인
        if str(model_path).endswith('.pth'):
            checkpoint_path = model_path
        else:
            checkpoint_path = Path(model_path) / "checkpoint.pth"
            
        if not Path(checkpoint_path).exists():
            LOGGER.error(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
            return {}
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Actor 네트워크에서 파라미터 추론 (config는 무시)
        if 'actor_state_dict' in checkpoint:
            actor_state_dict = checkpoint['actor_state_dict']
            
            # fc1.weight의 shape에서 state_dim과 hidden_dim 추론
            if 'fc1.weight' in actor_state_dict:
                fc1_weight_shape = actor_state_dict['fc1.weight'].shape
                hidden_dim = fc1_weight_shape[0]
                state_dim = fc1_weight_shape[1]
                
                LOGGER.info(f"actor_state_dict에서 추론된 파라미터: state_dim={state_dim}, hidden_dim={hidden_dim}")
                return {'state_dim': state_dim, 'hidden_dim': hidden_dim}
        
        LOGGER.warning("체크포인트에서 파라미터를 추론할 수 없습니다.")
        return {}
        
    except Exception as e:
        LOGGER.error(f"체크포인트에서 파라미터 추론 실패: {str(e)}")
        import traceback
        LOGGER.error(f"상세 오류:\n{traceback.format_exc()}")
        return {}


def load_model(args, model_path, env, device):
    """
    학습된 모델 로드
    """
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 환경에서 상태 차원 계산 (참고용)
        env_state_dim = calculate_state_dim(env, args.use_cnn)
        LOGGER.info(f"환경에서 계산된 상태 차원: {env_state_dim}")
        
        # 파라미터 설정 (우선순위: 명령행 인자 > actor_state_dict에서 추론 > 환경에서 계산)
        if args.state_dim is not None and args.hidden_dim is not None:
            # 명령행 인자가 모두 지정된 경우 최우선 사용
            final_state_dim = args.state_dim
            final_hidden_dim = args.hidden_dim
            LOGGER.info(f"명령행 인자 사용: state_dim={final_state_dim}, hidden_dim={final_hidden_dim}")
        else:
            # 체크포인트에서 모델 파라미터 추론 (config는 무시)
            checkpoint_params = infer_model_params_from_checkpoint(model_path)
            LOGGER.info(f"체크포인트에서 추론된 파라미터: {checkpoint_params}")
            
            # state_dim 설정
            if args.state_dim is not None:
                final_state_dim = args.state_dim
                LOGGER.info(f"state_dim은 명령행 인자 사용: {final_state_dim}")
            elif checkpoint_params and 'state_dim' in checkpoint_params:
                final_state_dim = checkpoint_params['state_dim']
                LOGGER.info(f"state_dim은 체크포인트에서 추론: {final_state_dim}")
            else:
                final_state_dim = env_state_dim
                LOGGER.warning(f"state_dim은 환경에서 계산 사용: {final_state_dim}")
            
            # hidden_dim 설정
            if args.hidden_dim is not None:
                final_hidden_dim = args.hidden_dim
                LOGGER.info(f"hidden_dim은 명령행 인자 사용: {final_hidden_dim}")
            elif checkpoint_params and 'hidden_dim' in checkpoint_params:
                final_hidden_dim = checkpoint_params['hidden_dim']
                LOGGER.info(f"hidden_dim은 체크포인트에서 추론: {final_hidden_dim}")
            else:
                final_hidden_dim = 256  # 기본값
                LOGGER.warning(f"hidden_dim은 기본값 사용: {final_hidden_dim}")
            
        LOGGER.info(f"최종 모델 파라미터: state_dim={final_state_dim}, hidden_dim={final_hidden_dim}, action_dim={args.action_dim}")
            
        # SAC 에이전트 초기화
        agent = SACAgent(
            state_dim=final_state_dim,
            action_dim=args.action_dim,
            hidden_dim=final_hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            alpha_lr=args.alpha_lr,
            gamma=args.gamma,
            tau=args.tau,
            alpha_init=args.alpha_init,
            target_update_interval=args.target_update_interval,
            use_automatic_entropy_tuning=args.use_automatic_entropy_tuning,
            buffer_capacity=args.buffer_capacity,
            use_cnn=args.use_cnn,
            use_gpu_memory=args.use_gpu_memory,
            device=device
        )
        
        # 모델 가중치 로드
        agent.load_model(model_path)
        
        # 모델을 evaluation 모드로 설정
        agent.actor.eval()
        if hasattr(agent, 'critic'):
            agent.critic.eval()
        if hasattr(agent, 'critic_target'):
            agent.critic_target.eval()
        
        LOGGER.info(f"모델 로드 완료: {model_path}")
        return agent
        
    except Exception as e:
        LOGGER.error(f"모델 로드 중 오류 발생: {str(e)}")
        raise


def load_data(data_path):
    """
    테스트 데이터 로드
    
    Args:
        data_path: 데이터 파일 경로
        
    Returns:
        로드된 데이터 DataFrame
    """
    try:
        # 파일 확장자 확인
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.h5'):
            data = pd.read_hdf(data_path)
        else:
            print(f"지원되지 않는 파일 형식: {data_path}")
            sys.exit(1)
            
        print(f"테스트 데이터 로드 완료: {len(data)} 행")
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        sys.exit(1)


def load_benchmark_data(benchmark_path):
    """
    벤치마크 데이터 로드
    
    Args:
        benchmark_path: 벤치마크 데이터 파일 경로
        
    Returns:
        로드된 벤치마크 수익률 배열
    """
    try:
        if benchmark_path is None:
            return None
            
        # 파일 확장자 확인
        if benchmark_path.endswith('.csv'):
            data = pd.read_csv(benchmark_path)
        elif benchmark_path.endswith('.parquet'):
            data = pd.read_parquet(benchmark_path)
        elif benchmark_path.endswith('.h5'):
            data = pd.read_hdf(benchmark_path)
        else:
            print(f"지원되지 않는 벤치마크 파일 형식: {benchmark_path}")
            return None
            
        # 수익률 컬럼 확인 및 추출
        if 'return' in data.columns:
            returns = data['return'].values
        elif 'returns' in data.columns:
            returns = data['returns'].values
        elif 'daily_return' in data.columns:
            returns = data['daily_return'].values
        else:
            print("벤치마크 데이터에서 수익률 컬럼을 찾을 수 없습니다.")
            return None
            
        print(f"벤치마크 데이터 로드 완료: {len(returns)} 행")
        return returns
    except Exception as e:
        print(f"벤치마크 데이터 로드 중 오류 발생: {e}")
        return None


def setup_logger(results_dir):
    """
    로거 설정
    
    Args:
        results_dir: 결과 저장 디렉토리
        
    Returns:
        로거 객체
    """
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    return Logger(log_file)


def load_test_data(symbols: list, args) -> Dict[str, pd.DataFrame]:
    """테스트 데이터 로드"""
    test_data = {}
    processed_dir = Path(args.data_path) / "processed"
    
    for symbol in symbols:
        test_data_path = processed_dir / symbol / "test_data.csv"
        
        if not test_data_path.exists():
            LOGGER.error(f"{symbol}의 테스트 데이터 파일이 없습니다: {test_data_path}")
            continue
            
        try:
            test_data[symbol] = pd.read_csv(test_data_path)
            LOGGER.info(f"{symbol} 테스트 데이터 로드 완료: {test_data[symbol].shape}")
        except Exception as e:
            LOGGER.error(f"{symbol} 테스트 데이터 로드 실패: {str(e)}")
    
    return test_data


def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"백테스팅 시작: 대상 심볼 {symbols}")
    
    # 테스트 데이터 로드
    test_data = load_test_data(symbols, args)
    
    if not test_data:
        LOGGER.error("테스트 데이터 로드 실패")
        return
    
    # 데이터 전처리 (타입 변환 및 날짜 컬럼 제거)
    for symbol, data in test_data.items():
        if "timestamp" in data.columns:
            data = data.drop(columns=["timestamp"])
        test_data[symbol] = data.astype(float)
    
    # 설정값 가져오기
    window_size = args.window_size if args.window_size is not None else config.WINDOW_SIZE
    max_trading_units = getattr(config, 'MAX_TRADING_UNITS', 1)
    initial_balance = args.initial_balance if args.initial_balance is not None else config.INITIAL_BALANCE
    transaction_fee_percent = args.transaction_fee_percent if args.transaction_fee_percent is not None else config.TRANSACTION_FEE_PERCENT
    
    # 첫 번째 심볼의 데이터로 임시 환경 생성 (모델 파라미터 추론용)
    first_symbol = list(test_data.keys())[0]
    temp_env = TradingEnvironment(
        data=test_data[first_symbol],
        window_size=window_size,
        initial_balance=initial_balance,
        max_trading_units=max_trading_units,
        transaction_fee_percent=transaction_fee_percent,
        symbol=first_symbol
    )
    
    # 모델 로드
    model_path = args.model_path
    if not model_path:
        LOGGER.error("모델 경로가 지정되지 않았습니다.")
        return
        
    try:
        agent = load_model(args, model_path, temp_env, device=DEVICE)
        LOGGER.info("모델 로드 완료")
    except Exception as e:
        LOGGER.error(f"모델 로드 실패: {str(e)}")
        return
    
    # 백테스팅 실행
    for symbol, data in test_data.items():
        try:
            LOGGER.info(f"{symbol} 백테스팅 시작")
            
            # 환경 생성
            env = TradingEnvironment(
                data=data,
                window_size=window_size,
                initial_balance=initial_balance,
                max_trading_units=max_trading_units,
                transaction_fee_percent=transaction_fee_percent,
                symbol=symbol
            )
            
            # 백테스터 생성 및 실행
            backtester = Backtester(
                agent=agent,
                test_data=data,
                config=args,
                logger=LOGGER,
                initial_balance=initial_balance,
                transaction_fee_percent=transaction_fee_percent,
            )
            
            results = backtester.run()
            
            # backtester 내부에 results를 설정
            backtester.results = results 
        
            # 결과 저장
            save_path = Path(args.output_dir) / f"backtest_{symbol}_{get_timestamp()}.json"
            backtester.save_results(save_path)
            
            LOGGER.info(f"{symbol} 백테스팅 완료: 결과 저장 경로 {save_path}")
            
        except Exception as e:
            LOGGER.error(f"{symbol} 백테스팅 중 오류 발생: {str(e)}")
            import traceback
            LOGGER.error(f"상세 오류:\n{traceback.format_exc()}")
            continue


if __name__ == "__main__":
    main()
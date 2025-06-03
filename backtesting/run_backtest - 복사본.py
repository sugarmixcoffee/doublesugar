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
from src.config.config import Config
from src.backtesting.backtester import Backtester
from src.backtesting.visualizer import Visualizer

from src.config.config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER,
    DATA_DIR
)
def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 모델 백테스트 실행')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델의 경로')
    parser.add_argument('--config_path', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--data_path', type=str, required=True,
                        help='테스트 데이터 경로')
    parser.add_argument('--results_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='초기 자본금')
    parser.add_argument('--commission_rate', type=float, default=0.0025,
                        help='거래 수수료율 (0.0025 = 0.25%)')
    parser.add_argument('--benchmark_data_path', type=str, default=None,
                        help='벤치마크 데이터 경로 (옵션)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='관측 창 크기 (설정에서 가져오지 않을 경우)')
    parser.add_argument('--symbols', nargs='+', help='대상 심볼 목록')
    parser.add_argument('--output_dir', type=str, default='results/backtest',
                        help='결과 저장 디렉토리')
    
    return parser.parse_args()


def load_model(model_path, config, device):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        config: 설정 객체
        device: 사용할 장치 (CPU/GPU)
        
    Returns:
        로드된 SAC 에이전트
    """
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # SAC 에이전트 초기화
        agent = SACAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            actor_learning_rate=config.actor_lr,
            critic_learning_rate=config.critic_lr,
            alpha_learning_rate=config.alpha_lr,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=config.batch_size,
            device=device,
            initial_alpha=config.initial_alpha,
            target_entropy=config.target_entropy
        )
        
        # 저장된 모델 로드
        agent.load_model(model_path)
        
        # 모델을 evaluation 모드로 설정
        agent.actor.eval()
        agent.critic1.eval()
        agent.critic2.eval()
        
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return agent
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        sys.exit(1)


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


def load_test_data(symbols: list) -> Dict[str, pd.DataFrame]:
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
    test_data = load_test_data(symbols)
    
    if not test_data:
        LOGGER.error("테스트 데이터 로드 실패")
        return
    
    # 모델 로드
    model_path = args.model_path
    if not model_path:
        LOGGER.error("모델 경로가 지정되지 않았습니다.")
        return
        
    try:
        agent = load_model(model_path, args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        LOGGER.info("모델 로드 완료")
    except Exception as e:
        LOGGER.error(f"모델 로드 실패: {str(e)}")
        return
    
    # 백테스팅 실행
    for symbol, data in test_data.items():
        try:
            # 환경 생성
            env = TradingEnvironment(
                data=data,
                window_size=args.window_size,
                initial_balance=args.initial_balance
            )
            
            # 백테스팅 실행
            backtester = Backtester(
                agent=agent,
                test_data=data,
                config=args,
                initial_balance=args.initial_balance
            )
            
            results = backtester.run()
            
            # 결과 저장
            save_path = Path(args.output_dir) / f"backtest_{symbol}_{get_timestamp()}.json"
            backtester.save_results(save_path)
            
            LOGGER.info(f"{symbol} 백테스팅 완료: 결과 저장 경로 {save_path}")
            
        except Exception as e:
            LOGGER.error(f"{symbol} 백테스팅 중 오류 발생: {str(e)}")
            continue


if __name__ == "__main__":
    main()
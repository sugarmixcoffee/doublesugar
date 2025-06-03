"""
SAC 모델 학습 실행 스크립트 - 최적화된 버전
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import psutil
from typing import Dict, Optional, Tuple, Any
import logging
from contextlib import contextmanager

from src.config.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    TARGET_SYMBOLS,
    DATA_DIR,
    LOGGER,
    USE_GPU_MEMORY,
    VRAM_PREFETCH,
    GPU_BATCH_SIZE,
    WINDOW_SIZE,
    TRAIN_RATIO,
    TEST_RATIO,
    TRANSACTION_FEE_PERCENT,
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    USE_MIXED_PRECISION,
    REWARD_TYPE
)
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.models.sac_agent import SACAgent
from src.training.trainer import Trainer
from src.utils.utils import create_directory, get_timestamp, load_from_csv
from src.preprocessing.normalize_data import DivideNormalization, Sampling_data

# 메모리 관리 설정
MEMORY_CONFIG = {
    'gpu_memory_threshold': 0.8,
    'chunk_size': 50000,
    'cleanup_interval': 10,
    'prefetch_enabled': True,
    'memory_check_frequency': 5  # 5번마다 한 번 체크
}

class DataLoadError(Exception):
    """데이터 로딩 관련 예외"""
    pass

class ModelInitError(Exception):
    """모델 초기화 관련 예외"""
    pass

class MemoryManager:
    """메모리 관리 클래스"""
    
    def __init__(self):
        self.cleanup_counter = 0
        
    def check_gpu_memory_threshold(self, threshold: float = 0.8) -> bool:
        """GPU 메모리 임계값 체크"""
        if not torch.cuda.is_available():
            return False
        
        current = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return current > threshold
    
    def cleanup_memory_if_needed(self, force: bool = False) -> None:
        """필요시 메모리 정리"""
        self.cleanup_counter += 1
        
        if force or self.cleanup_counter % MEMORY_CONFIG['cleanup_interval'] == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup_counter = 0
    
    def get_memory_info(self) -> str:
        """메모리 사용량 정보 반환"""
        info_parts = []
        
        # GPU 메모리
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info_parts.append(f"GPU: {allocated:.2f}/{total:.2f}GB")
        
        # RAM 메모리
        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024**3)
        info_parts.append(f"RAM: {ram_usage:.2f}GB")
        
        return " | ".join(info_parts)

@contextmanager
def memory_profiler(operation_name: str):
    """메모리 사용량 프로파일링 컨텍스트 매니저"""
    memory_manager = MemoryManager()
    
    before = memory_manager.get_memory_info()
    LOGGER.info(f"{operation_name} 시작 - 메모리: {before}")
    
    try:
        yield
    finally:
        after = memory_manager.get_memory_info()
        LOGGER.info(f"{operation_name} 완료 - 메모리: {after}")

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 학습 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None, help='학습에 사용할 주식 심볼 목록')
    
    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=30, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='초기 자본금')
    parser.add_argument('--multi_asset', action='store_true', help='다중 자산 환경 사용 여부')
    
    # 모델 관련 인자
    parser.add_argument('--hidden_dim', type=int, default=256, help='은닉층 차원')
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용 여부')
    parser.add_argument('--load_model', type=str, default=None, help='로드할 모델 경로')
    
    # 학습 관련 인자
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--num_episodes', type=int, default=NUM_EPISODES, help='학습할 총 에피소드 수')
    parser.add_argument('--evaluate_interval', type=int, default=EVALUATE_INTERVAL, help='평가 간격')
    parser.add_argument('--save_interval', type=int, default=SAVE_MODEL_INTERVAL, help='모델 저장 간격')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='에피소드당 최대 스텝 수')
    
    return parser.parse_args()

def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """데이터 청크 전처리"""
    # 날짜 컬럼 패턴
    date_patterns = ['date', 'time', 'timestamp', 'dt', 'day', 'month', 'year']
    
    # 날짜 및 문자열 컬럼 제거
    columns_to_drop = []
    for col in chunk.columns:
        # 열 이름에 날짜 패턴이 있거나 문자열 타입인 경우 제거
        if any(pattern in col.lower() for pattern in date_patterns) or chunk[col].dtype == 'object':
            columns_to_drop.append(col)
            
    if columns_to_drop:
        LOGGER.info(f"제거할 열: {columns_to_drop}")
        chunk = chunk.drop(columns=columns_to_drop)
    
    # 비숫자 열 감지 및 제거
    non_numeric_cols = chunk.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        LOGGER.info(f"추가 비숫자 열 제거: {non_numeric_cols}")
        chunk = chunk.drop(columns=non_numeric_cols)
    
    # 타입 최적화
    chunk = chunk.astype(np.float32)
    
    # 무한대 값 처리
    chunk.replace([np.inf, -np.inf], 0, inplace=True)
    
    return chunk

def load_single_symbol_data(symbol: str, processed_dir: Path, 
                           memory_manager: MemoryManager) -> Optional[pd.DataFrame]:
    """단일 심볼 데이터 로드"""
    normalized_data_path = processed_dir / symbol / "normalized_data.csv"
    
    if not normalized_data_path.exists():
        LOGGER.error(f"{symbol}의 전처리된 데이터 파일이 없습니다: {normalized_data_path}")
        return None
    
    try:
        # 파일 크기 확인
        file_size = os.path.getsize(normalized_data_path) / (1024**3)  # GB
        LOGGER.info(f"{symbol} 데이터 파일 크기: {file_size:.2f}GB")
        
        # GPU 사용 가능성 및 메모리 체크
        use_gpu = USE_GPU_MEMORY and torch.cuda.is_available()
        gpu_mem_available = 0
        
        if use_gpu:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_available = total_mem * 0.5  # 50% 임계값
        
        # 데이터 로딩 전략 결정
        if use_gpu and file_size > gpu_mem_available:
            # 대용량 파일: 청크 단위 처리
            LOGGER.info(f"{symbol} 데이터가 크므로 청크 단위로 처리합니다.")
            normalized_data = load_large_file_chunked(normalized_data_path, memory_manager)
        else:
            # 일반 크기 파일: 한 번에 로드
            normalized_data = load_regular_file(normalized_data_path, use_gpu)
        
        if normalized_data is not None:
            # 인덱스 정리
            if isinstance(normalized_data.index, (pd.DatetimeIndex, pd.Index)):
                if normalized_data.index.dtype == 'object':
                    normalized_data = normalized_data.reset_index(drop=True)
            
            LOGGER.info(f"{symbol} 데이터 로드 완료: {normalized_data.shape}")
            memory_manager.cleanup_memory_if_needed()
            
        return normalized_data
        
    except Exception as e:
        error_type = handle_data_load_error(symbol, e)
        if error_type == "memory_error":
            # 메모리 부족 시 강제 정리 후 재시도
            memory_manager.cleanup_memory_if_needed(force=True)
            return None
        else:
            LOGGER.error(f"{symbol} 데이터 로드 실패: {str(e)}")
            return None

def load_large_file_chunked(file_path: Path, memory_manager: MemoryManager) -> pd.DataFrame:
    """대용량 파일을 청크 단위로 로드"""
    chunk_size = MEMORY_CONFIG['chunk_size']
    chunks = []
    
    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            processed_chunk = preprocess_chunk(chunk)
            chunks.append(processed_chunk)
            
            # 주기적 메모리 체크
            if i % MEMORY_CONFIG['memory_check_frequency'] == 0:
                if memory_manager.check_gpu_memory_threshold():
                    LOGGER.warning("GPU 메모리 사용량이 높습니다. 정리를 수행합니다.")
                    memory_manager.cleanup_memory_if_needed(force=True)
        
        # 모든 청크 결합
        normalized_data = pd.concat(chunks, ignore_index=True)
        del chunks
        memory_manager.cleanup_memory_if_needed()
        
        return normalized_data
        
    except Exception as e:
        LOGGER.error(f"청크 단위 로딩 실패: {str(e)}")
        return None

def load_regular_file(file_path: Path, use_gpu: bool) -> pd.DataFrame:
    """일반 크기 파일 로드"""
    try:
        # 먼저 혼합 타입으로 로드 (날짜/문자열 포함)
        normalized_data = pd.read_csv(
            file_path,
            memory_map=not use_gpu,  # GPU 사용 시 메모리 매핑 비활성화
            low_memory=True
        )
        
        # 전처리 (날짜 및 문자열 컬럼 제거)
        normalized_data = preprocess_chunk(normalized_data)
        
        return normalized_data
        
    except Exception as e:
        LOGGER.error(f"일반 파일 로딩 실패: {str(e)}")
        return None

def handle_data_load_error(symbol: str, error: Exception) -> str:
    """데이터 로딩 에러 전용 핸들러"""
    error_str = str(error).lower()
    
    if "memory" in error_str or "out of memory" in error_str:
        LOGGER.error(f"{symbol}: 메모리 부족으로 로딩 실패")
        return "memory_error"
    elif "file" in error_str or "no such file" in error_str:
        LOGGER.error(f"{symbol}: 파일 접근 오류")
        return "file_error"
    elif "permission" in error_str:
        LOGGER.error(f"{symbol}: 파일 권한 오류")
        return "permission_error"
    else:
        LOGGER.error(f"{symbol}: 알 수 없는 오류 - {error}")
        return "unknown_error"

def check_and_prepare_data(symbol: str, memory_manager: MemoryManager) -> Optional[pd.DataFrame]:
    """
    데이터 파일 존재 여부 확인 및 필요시 전처리 수행
    
    Args:
        symbol: 주식 심볼
        memory_manager: 메모리 관리자
        
    Returns:
        전처리된 데이터 DataFrame
    """
    LOGGER.info(f"{symbol} 데이터 준비 시작")
    processed_dir = DATA_DIR / "processed" / symbol
    train_data_path = processed_dir / "train_data.csv"
    
    # train_data.csv가 없으면 전처리 수행
    if not train_data_path.exists():
        LOGGER.info(f"{symbol}의 train_data.csv가 없습니다. 데이터 전처리를 시작합니다.")
        
        # 원본 데이터 경로
        raw_data_path = DATA_DIR / f"{symbol}.csv"
        if not raw_data_path.exists():
            LOGGER.error(f"{symbol}의 원본 데이터 파일이 없습니다: {raw_data_path}")
            return None
            
        try:
            # 원본 데이터 로드
            LOGGER.info(f"{symbol} 원본 데이터 로드 시도: {raw_data_path}")
            raw_data = load_from_csv(raw_data_path)
            if raw_data is None:
                LOGGER.error(f"{symbol}의 원본 데이터 로드 실패")
                return None
            LOGGER.info(f"{symbol} 원본 데이터 로드 완료: {raw_data.shape}")
                
            # 데이터 전처리
            LOGGER.info(f"{symbol} 데이터 전처리 시작")
            try:
                normalizer = DivideNormalization()
                processed_data = normalizer.divide_data(
                    data=raw_data,
                    symbol=symbol,
                    
                    train_ratio=TRAIN_RATIO,
                    test_ratio=TEST_RATIO,
                    window_size=WINDOW_SIZE
                )
                
                if processed_data is None or 'normalized_data' not in processed_data:
                    LOGGER.error(f"{symbol} 데이터 전처리 결과가 올바르지 않습니다")
                    return None
                    
                LOGGER.info(f"{symbol} 데이터 전처리 완료")
                return processed_data['normalized_data']
                
            except Exception as e:
                LOGGER.error(f"{symbol} 데이터 전처리 중 오류 발생: {str(e)}")
                LOGGER.error(f"전처리 오류 상세: {e.__class__.__name__}")
                import traceback
                LOGGER.error(f"전처리 오류 스택트레이스:\n{traceback.format_exc()}")
                return None
            
        except Exception as e:
            LOGGER.error(f"{symbol} 원본 데이터 로드 중 오류 발생: {str(e)}")
            LOGGER.error(f"로드 오류 상세: {e.__class__.__name__}")
            import traceback
            LOGGER.error(f"로드 오류 스택트레이스:\n{traceback.format_exc()}")
            return None
    
    # train_data.csv가 있으면 로드 시도
    try:
        # 파일 크기 확인
        file_size = os.path.getsize(train_data_path) / (1024**3)  # GB
        LOGGER.info(f"{symbol} 데이터 파일 크기: {file_size:.2f}GB")
        
        # GPU 사용 가능성 및 메모리 체크
        use_gpu = USE_GPU_MEMORY and torch.cuda.is_available()
        gpu_mem_available = 0
        
        if use_gpu:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_available = total_mem * 0.5  # 50% 임계값
        
        # 데이터 로딩 전략 결정
        if use_gpu and file_size > gpu_mem_available:
            # 대용량 파일: 청크 단위 처리
            LOGGER.info(f"{symbol} 데이터가 크므로 청크 단위로 처리합니다.")
            normalized_data = load_large_file_chunked(train_data_path, memory_manager)
        else:
            # 일반 크기 파일: 한 번에 로드
            normalized_data = load_regular_file(train_data_path, use_gpu)
        
        if normalized_data is not None:
            # 인덱스 정리
            if isinstance(normalized_data.index, (pd.DatetimeIndex, pd.Index)):
                if normalized_data.index.dtype == 'object':
                    normalized_data = normalized_data.reset_index(drop=True)
            
            LOGGER.info(f"{symbol} 데이터 로드 완료: {normalized_data.shape}")
            memory_manager.cleanup_memory_if_needed()
            return normalized_data
        else:
            LOGGER.warning(f"{symbol} 데이터 로드 실패. 전처리를 다시 수행합니다.")
            # 데이터 로드 실패 시 전처리 수행
            raw_data_path = DATA_DIR / f"{symbol}.csv"
            if not raw_data_path.exists():
                LOGGER.error(f"{symbol}의 원본 데이터 파일이 없습니다: {raw_data_path}")
                return None
                
            try:
                # 원본 데이터 로드
                LOGGER.info(f"{symbol} 원본 데이터 로드 시도: {raw_data_path}")
                raw_data = load_from_csv(raw_data_path)
                if raw_data is None:
                    LOGGER.error(f"{symbol}의 원본 데이터 로드 실패")
                    return None
                LOGGER.info(f"{symbol} 원본 데이터 로드 완료: {raw_data.shape}")
                    
                # 데이터 전처리
                LOGGER.info(f"{symbol} 데이터 전처리 시작")
                try:
                    normalizer = DivideNormalization()
                    processed_data = normalizer.divide_data(
                        data=raw_data,
                        symbol=symbol,
                        train_ratio=TRAIN_RATIO,
                        test_ratio=TEST_RATIO,
                        window_size=WINDOW_SIZE
                    )
                    
                    if processed_data is None or 'normalized_data' not in processed_data:
                        LOGGER.error(f"{symbol} 데이터 전처리 결과가 올바르지 않습니다")
                        return None
                        
                    LOGGER.info(f"{symbol} 데이터 전처리 완료")
                    return processed_data['normalized_data']
                    
                except Exception as e:
                    LOGGER.error(f"{symbol} 데이터 전처리 중 오류 발생: {str(e)}")
                    LOGGER.error(f"전처리 오류 상세: {e.__class__.__name__}")
                    import traceback
                    LOGGER.error(f"전처리 오류 스택트레이스:\n{traceback.format_exc()}")
                    return None
                
            except Exception as e:
                LOGGER.error(f"{symbol} 원본 데이터 로드 중 오류 발생: {str(e)}")
                LOGGER.error(f"로드 오류 상세: {e.__class__.__name__}")
                import traceback
                LOGGER.error(f"로드 오류 스택트레이스:\n{traceback.format_exc()}")
                return None
            
    except Exception as e:
        LOGGER.error(f"{symbol} 데이터 로드 중 오류 발생: {str(e)}")
        LOGGER.error(f"로드 오류 상세: {e.__class__.__name__}")
        import traceback
        LOGGER.error(f"로드 오류 스택트레이스:\n{traceback.format_exc()}")
        return None


def load_processed_data(symbols: list) -> Dict[str, pd.DataFrame]:
    """ 전처리된 데이터 로드 - 최적화된 버전
        없을 경우 데이터를 분리, 전처리해서 로딩한다. 
    """
    processed_data = {}
    processed_dir = DATA_DIR / "processed"
    memory_manager = MemoryManager()
    
    # GPU 초기화 및 상태 확인
    if torch.cuda.is_available() and USE_GPU_MEMORY:
        LOGGER.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        LOGGER.info(f"초기 메모리 상태: {memory_manager.get_memory_info()}")
    
    # 각 심볼별 데이터 로드
    for symbol in symbols:
        with memory_profiler(f"{symbol} 데이터 로딩"):
            train_data_path = processed_dir / symbol / "train_data.csv"
            
            # train 데이터가 없을경우: 원본을 로딩해서 전처리후 저장한다. 
            if not train_data_path.exists():
                LOGGER.error(f"{symbol}의 학습 데이터 파일이 없습니다: {train_data_path}")
                data_dir = DATA_DIR
                csv_file = f"{symbol}.csv"
                file_path = os.path.join(data_dir, csv_file)
                try: 
                    # 원본 파일 로딩
                    # 파일 존재 여부 확인
                    if not os.path.exists(file_path):
                        LOGGER.info(f"{symbol} raw 데이터 파일이 존재하지 않습니다.")
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)    
                    LOGGER.info(f"{symbol} raw 데이터 로드 완료, 데이터 크기: {df.shape}")
                    if df.empty:
                        LOGGER.error(f"raw 데이터가 비어있습니다: {file_path}")
                        
                    # 최근 데이터 샘플링
                    sample = Sampling_data()
                    sampled_df = sample.small_data_sample(df)
                    LOGGER.info(f"샘플링 후 데이터 크기: {sampled_df.shape}")

                    # 데이터 분리 및 정규화
                    divide_df = DivideNormalization()
                    result = divide_df.divide_data(sampled_df, symbol)
                    
                except Exception as e:
                    LOGGER.error(f"{symbol} raw 데이터 로드 실패: {str(e)}")  
                    return None
            # train 데이터가 있을 경우 해당파일을 로딩만 한다.      
            else: 
                #     # train 로딩
                try:
                #     if not os.path.exists(train_data_path):
                #         print(f"{symbol} raw 데이터 파일이 존재하지 않습니다.")
                #     df = pd.read_csv(file_path, index_col=0, parse_dates=True)    
                #     print(f"{symbol} raw 데이터 로드 완료, 데이터 크기: {df.shape}")
                    # 파일 크기 확인
                    file_size = os.path.getsize(train_data_path) / (1024**3)  # GB
                    LOGGER.info(f"{symbol} 학습 데이터 파일 크기: {file_size:.2f}GB")
                    
                    # GPU 사용 가능성 및 메모리 체크
                    use_gpu = USE_GPU_MEMORY and torch.cuda.is_available()
                    gpu_mem_available = 0
                    
                    if use_gpu:
                        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_mem_available = total_mem * 0.5  # 50% 임계값
                    
                    # 데이터 로딩 전략 결정
                    if use_gpu and file_size > gpu_mem_available:
                        # 대용량 파일: 청크 단위 처리
                        LOGGER.info(f"{symbol} 데이터가 크므로 청크 단위로 처리합니다.")
                        train_data = load_large_file_chunked(train_data_path, memory_manager)
                    else:
                        # 일반 크기 파일: 한 번에 로드
                        train_data = load_regular_file(train_data_path, use_gpu)
                    
                    if train_data is not None:
                        # 인덱스 정리
                        if isinstance(train_data.index, (pd.DatetimeIndex, pd.Index)):
                            if train_data.index.dtype == 'object':
                                train_data = train_data.reset_index(drop=True)
                        
                        LOGGER.info(f"{symbol} 학습 데이터 로드 완료: {train_data.shape}")
                        processed_data[symbol] = train_data
                        memory_manager.cleanup_memory_if_needed()
                        
                except Exception as e:
                    error_type = handle_data_load_error(symbol, e)
                    if error_type == "memory_error":
                        # 메모리 부족 시 강제 정리 후 재시도
                        memory_manager.cleanup_memory_if_needed(force=True)
                    else:
                        LOGGER.error(f"{symbol} 데이터 로드 실패: {str(e)}")
     
    LOGGER.info(f"총 {len(processed_data)}개 심볼 데이터 로드 완료")
    LOGGER.info(f"최종 메모리 상태: {memory_manager.get_memory_info()}")
    
    return processed_data
# def load_processed_data(symbols: list) -> Dict[str, pd.DataFrame]:
#     """
#     전처리된 데이터 로드
    
#     Args:
#         symbols: 주식 심볼 목록
        
#     Returns:
#         심볼별 데이터 딕셔너리
#     """
#     memory_manager = MemoryManager()
#     normalized_data_dict = {}
    
#     for symbol in symbols:
#         try:
#             # 데이터 확인 및 준비
#             normalized_data = check_and_prepare_data(symbol, memory_manager)
#             if normalized_data is not None and not normalized_data.empty:
#                 normalized_data_dict[symbol] = normalized_data
#                 LOGGER.info(f"{symbol} 데이터 준비 완료: {normalized_data.shape}")
#             else:
#                 LOGGER.error(f"{symbol} 데이터 준비 실패: 데이터가 비어있거나 None입니다")
#                 continue
                
#             # 메모리 관리
#             memory_manager.cleanup_memory_if_needed()
            
#         except Exception as e:
#             LOGGER.error(f"{symbol} 데이터 처리 중 예외 발생: {str(e)}")
#             continue
    
#     if not normalized_data_dict:
#         LOGGER.error("모든 심볼의 데이터 준비가 실패했습니다")
#     else:
#         LOGGER.info(f"총 {len(normalized_data_dict)}개 심볼의 데이터 준비 완료")
    
#     return normalized_data_dict

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
            LOGGER.info(f"추정 상태 차원 사용: {estimated_dim}")
            return estimated_dim
        else:
            default_dim = 4322  # fallback
            LOGGER.info(f"기본 상태 차원 사용: {default_dim}")
            return default_dim

def create_environment(args, normalized_data_dict: Dict[str, pd.DataFrame], 
                      symbols: list) -> Tuple[Any, int]:
    """환경 생성"""
    try:
        if args.multi_asset:
            # 다중 자산 환경
            LOGGER.info("다중 자산 트레이딩 환경 생성 중...")
            env = MultiAssetTradingEnvironment(
                data_dict=normalized_data_dict,
                window_size=args.window_size,
                initial_balance=args.initial_balance
            )
            action_dim = len(symbols)
        else:
            # 단일 자산 환경
            symbol = symbols[0]
            LOGGER.info(f"단일 자산 트레이딩 환경 생성 중: {symbol}")
            
            if symbol not in normalized_data_dict:
                raise ValueError(f"{symbol} 데이터가 없습니다.")
            
            normalized_data = normalized_data_dict[symbol]
            env = TradingEnvironment(
                data=normalized_data,
                window_size=args.window_size,
                initial_balance=args.initial_balance,
                symbol=symbol,
                reward_type=REWARD_TYPE
            )
            action_dim = 1
            
            # 메모리 최적화: 사용하지 않는 데이터 해제
            for sym in list(normalized_data_dict.keys()):
                if sym != symbol:
                    del normalized_data_dict[sym]
            gc.collect()
            LOGGER.info("미사용 데이터 메모리 해제 완료")
        
        return env, action_dim
        
    except Exception as e:
        raise ModelInitError(f"환경 생성 실패: {str(e)}")

def create_agent(args, env, action_dim: int) -> SACAgent:
    """에이전트 생성"""
    try:
        if args.use_cnn:
            # CNN 모델
            if hasattr(env, 'feature_dim'):
                input_shape = (args.window_size, env.feature_dim)
            else:
                # 기본값 추정
                input_shape = (args.window_size, 144)  # 일반적인 feature 수
            
            LOGGER.info(f"CNN 에이전트 생성: input_shape={input_shape}")
            agent = SACAgent(
                action_dim=action_dim,
                hidden_dim=args.hidden_dim,
                input_shape=input_shape,
                use_cnn=True,
                use_gpu_memory=USE_GPU_MEMORY
            )
        else:
            # 일반 모델
            state_dim = calculate_state_dim(env, use_cnn=False)
            LOGGER.info(f"일반 에이전트 생성: state_dim={state_dim}")
            
            agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=args.hidden_dim,
                use_gpu_memory=USE_GPU_MEMORY
            )
        
        return agent
        
    except Exception as e:
        raise ModelInitError(f"에이전트 생성 실패: {str(e)}")

def initialize_gpu():
    """GPU 초기화 및 설정"""
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
        except:
            pass
        
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # GPU 정보 출력
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        LOGGER.info(f"GPU 초기화 완료: {device_name} ({total_memory:.1f}GB)")
        
        return True
    else:
        LOGGER.warning("GPU를 사용할 수 없어 CPU로 실행합니다.")
        return False

def main():
    """메인 함수 - 최적화된 버전"""
    
    symbols= ["AAPL", "BAC", "GS", "JNJ", "JPM", "MSFT", "NVDA"]   
    memory_manager = MemoryManager()
    train_data_dict = {}
    try:
        # GPU 초기화
        gpu_available = initialize_gpu()
        
        # 인자 파싱
        args = parse_args()
        symbols = args.symbols if args.symbols else TARGET_SYMBOLS
        LOGGER.info(f"학습 시작: 대상 심볼 {symbols}")
        
        # 종목별 데이터 로드
        
        # 데이터 로드 (한 번만 수행)
        with memory_profiler("전체 데이터 로딩"):
            train_data_dict = load_processed_data(symbols) # 훈련 데이터만 가져온다.
            
            if not train_data_dict:
                LOGGER.info(f"{symbol} 데이터 로드 중...")
                train_data = load_processed_data([symbols])
                raise DataLoadError("전처리된 데이터 로드 실패")
            
            # 데이터를 메모리에 유지하기 위해 복사
            train_data_dict = {symbol: data.copy() for symbol, data in train_data_dict.items()}
            LOGGER.info("데이터 로드 완료 및 메모리에 유지")
        
        # 환경 생성 (데이터 재사용)
        with memory_profiler("환경 생성"):
            if args.multi_asset:
                # 다중 자산 환경
                LOGGER.info("다중 자산 트레이딩 환경 생성 중...")
                env = MultiAssetTradingEnvironment(
                    data_dict=train_data_dict,
                    window_size=args.window_size,
                    initial_balance=args.initial_balance
                )
                action_dim = len(symbols)
            else:
                # 단일 자산 환경
                symbol = symbols[0]
                LOGGER.info(f"단일 자산 트레이딩 환경 생성 중: {symbol}")
                
                if symbol not in train_data_dict:
                    raise ValueError(f"{symbol} 데이터가 없습니다.")
                
                train_data = train_data_dict[symbol]
                env = TradingEnvironment(
                    data=train_data,
                    window_size=args.window_size,
                    initial_balance=args.initial_balance,
                    symbol=symbol,
                    reward_type=REWARD_TYPE
                )
                action_dim = 1
                
                # 메모리 최적화: 사용하지 않는 데이터 해제
                for sym in list(train_data_dict.keys()):
                    if sym != symbol:
                        del train_data_dict[sym]
                gc.collect()
                LOGGER.info("미사용 데이터 메모리 해제 완료")
        
        # 에이전트 생성
        with memory_profiler("에이전트 생성"):
            agent = create_agent(args, env, action_dim)
        
        # 모델 로드 (필요시)
        if args.load_model:
            try:
                LOGGER.info(f"모델 로드 중: {args.load_model}")
                agent.load_model(args.load_model)
                LOGGER.info("모델 로드 완료")
            except Exception as e:
                LOGGER.error(f"모델 로드 실패: {str(e)}")
                LOGGER.info("새 모델로 시작합니다.")
        
        # 트레이너 생성
        LOGGER.info("트레이너 생성 중...")
        trainer = Trainer(
            agent=agent,
            env=env,
            batch_size=args.batch_size,
            num_episodes=args.num_episodes,
            evaluate_interval=args.evaluate_interval,
            save_interval=args.save_interval,
            max_steps=args.max_steps
        )
        
        # 학습 실행
        LOGGER.info("학습 시작...")
        LOGGER.info(f"학습 시작 전 메모리: {memory_manager.get_memory_info()}")
        
        try:
            training_stats = trainer.train()
            
            final_reward = trainer.eval_rewards[-1] if trainer.eval_rewards else 'N/A'
            LOGGER.info(f"학습 완료: 최종 평가 보상 {final_reward}")
            LOGGER.info(f"학습 완료 후 메모리: {memory_manager.get_memory_info()}")
            
        except Exception as train_error:
            LOGGER.error(f"학습 중 오류 발생: {str(train_error)}")
            import traceback
            LOGGER.error(traceback.format_exc())
            raise
        
    except DataLoadError as e:
        LOGGER.error(f"데이터 로딩 오류: {str(e)}")
        return 1
        
    except ModelInitError as e:
        LOGGER.error(f"모델 초기화 오류: {str(e)}")
        return 1
        
    except Exception as e:
        LOGGER.error(f"예상치 못한 오류 발생: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return 1
    
    finally:
        # 최종 정리
        memory_manager.cleanup_memory_if_needed(force=True)
        if gpu_available:
            final_memory = memory_manager.get_memory_info()
            LOGGER.info(f"최종 메모리 상태: {final_memory}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
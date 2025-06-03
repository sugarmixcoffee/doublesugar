"""
SAC 트레이딩 시스템 설정 파일
"""
import os
import logging
import torch
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 데이터 관련 설정
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Alpha Vantage API 설정
ALPHA_VANTAGE_API_KEY = "D4ZHTU56ITGMY8F5"
API_CALL_DELAY = 12  # 초 단위 (API 호출 제한 고려)

# 대상 주식 종목 (미국 빅테크 기업)
TARGET_SYMBOLS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "GOOGL",  # Alphabet Inc. (Google)
    "AMZN",  # Amazon.com Inc.
    "META",  # Meta Platforms Inc. (Facebook)
    "NVDA",  # NVIDIA Corporation
    "TSLA",  # Tesla Inc.
    "NFLX",  # Netflix Inc.
    "INTC",  # Intel Corporation
    "AMD"    # Advanced Micro Devices Inc.
]

# 데이터 수집 설정
DATA_START_DATE = "2013-01-01"  # 10년 데이터
DATA_FREQUENCY = "daily"  # 일별 데이터

# 데이터 전처리 설정
WINDOW_SIZE = 30  # 관측 윈도우 크기
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 피쳐 설정
BASIC_FEATURES = [
    "open", "high", "low", "close", "volume"
]

MOMENTUM_FEATURES = [
    "rsi_7_1min", "rsi_14_1min", "rsi_21_1min",
    "macd_12_26_9_1min", "macd_signal_12_26_9_1min", "macd_hist_12_26_9_1min",
    "mfi_14_1min",
    "adx_20_1min",
    "roc_5_1min", "roc_20_1min",
    "cci_20_1min",
    "ultosc_5_15_30_1min"
]

TREND_FEATURES = [
    "ema_5_1min", "ema_21_1min", "ema_50_1min", 
    "sma_10_1min", "sma_20_1min", "sma_50_1min",
    "bband_upper_20_1min", "bband_middle_20_1min", "bband_lower_20_1min",
    "atr_14_1min", "atr_21_1min"
]

VOLUME_FEATURES = [
    "obv_1min",
    "close_diff_1"
]

PRICE_VARIATION_FEATURES = [
    "log_return_1", "close_pct_change_1"
]

# 트레이딩 환경 설정
INITIAL_BALANCE = 10000.0  # 초기 자본금
MAX_TRADING_UNITS = 10  # 최대 거래 단위
TRANSACTION_FEE_PERCENT = 0.001  # 거래 수수료 (0.1%)

# SAC 모델 하이퍼파라미터
HIDDEN_DIM = 256
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
LEARNING_RATE_ALPHA = 1e-4
GAMMA = 0.99
TAU = 0.005
ALPHA_INIT = 0.1
TARGET_UPDATE_INTERVAL = 2
REPLAY_BUFFER_SIZE = 500000
REPLAY_BUFFER_CLEANUP_FREQ = 1000

# 학습 설정
BATCH_SIZE = 1024
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50
MAX_STEPS_PER_EPISODE = 1000
MAX_EVAL_STEPS = 1000
MAX_EVAL_TIME = 300

# 장치 설정 - GPU 메모리(VRAM) 우선 사용
try:
    if torch.cuda.is_available():
        # 로거 초기화 전에 기본 로깅 사용
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        print(f"CUDA 버전: {torch.version.cuda}")
        DEVICE = torch.device("cuda:0")  # 첫 번째 GPU 장치 사용
        
        # GPU 메모리 정보 출력
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위
        print(f"GPU 총 메모리: {gpu_mem_total:.2f} GB")
        
        # CUDA 메모리 할당 관련 설정
        torch.backends.cudnn.benchmark = True  # 성능 최적화
        torch.backends.cudnn.deterministic = False  # 더 빠른 처리를 위해 비결정적 알고리즘 허용
        
        # GPU 메모리(VRAM) 설정
        USE_GPU_MEMORY = True
        MAX_GPU_MEMORY_USAGE = 0.9  # GPU 메모리 최대 사용 비율 (90%)
        
        # 메모리 사용량 최적화 (필요시 DRAM으로 스왑)
        ENABLE_MEMORY_SWAP = True
        MEMORY_FRACTION = 0.8  # 메모리 사용 비율 제한
        
        # 텐서 정밀도 설정
        USE_MIXED_PRECISION = True  # 혼합 정밀도 활성화 (FP16)
        DEFAULT_DTYPE = torch.float32  # 기본 텐서 타입
        
        # 메모리 할당 설정
        ALLOCATOR_CONFIG = {
            'max_split_size_mb': 128,  # 메모리 단편화 방지를 위한 최대 분할 크기
            'roundup_power2': True,    # 2의 제곱으로 메모리 크기 반올림
        }
        
        # 혼합 정밀도 설정 적용
        if USE_MIXED_PRECISION:
            torch.set_float32_matmul_precision('high')
    else:
        DEVICE = torch.device("cpu")
        print("CUDA를 사용할 수 없어 CPU로 실행합니다.")
        USE_GPU_MEMORY = False
        ENABLE_MEMORY_SWAP = False
        USE_MIXED_PRECISION = False
except Exception as e:
    print(f"GPU 초기화 중 오류 발생: {str(e)}")
    DEVICE = torch.device("cpu")
    USE_GPU_MEMORY = False
    ENABLE_MEMORY_SWAP = False
    USE_MIXED_PRECISION = False

# 메모리 관리 설정
MEMORY_CLEANUP_INTERVAL = 5  # 몇 번의 에피소드마다 메모리 정리를 수행할지
USE_MEMORY_MAPPING = not USE_GPU_MEMORY  # GPU 사용 시 메모리 매핑 비활성화
LOW_MEMORY_MODE = not USE_GPU_MEMORY  # GPU 사용 시 저메모리 모드 비활성화

# GPU 배치 최적화 설정
GPU_BATCH_SIZE = 512  # GPU 메모리에 최적화된 배치 크기
VRAM_PREFETCH = True  # 데이터 미리 로드 활성화

# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    """로거 설정 함수"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

# 기본 로거 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"sac_trading_{timestamp}.log"
LOGGER = setup_logger("sac_trading", LOG_FILE)

# 백테스트 설정
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2023-01-01"

# ===== 거래 장려 모드 설정 =====
# 거래를 더 활발하게 하도록 유도하는 설정
USE_TRADE_ENCOURAGING_MODE = False  # True로 설정하면 거래 장려 모드 활성화

# 거래 장려 모드에서 사용할 파라미터 (USE_TRADE_ENCOURAGING_MODE가 True일 때만 적용)
if USE_TRADE_ENCOURAGING_MODE:
    # 엔트로피 관련
    ALPHA_INIT = 0.5  # 더 높은 초기 엔트로피 (더 많은 탐색)
    LEARNING_RATE_ACTOR = 1e-4  # 안정성을 위해 학습률 조정
    LEARNING_RATE_ALPHA = 3e-5  # 엔트로피 학습률 감소
    
    # 거래 환경 설정
    MIN_TRADE_THRESHOLD = 0.05  # 낮은 거래 임계값
    MAX_NO_TRADE_STEPS = 20  # 거래하지 않으면 패널티 시작
    NO_TRADE_PENALTY_RATE = 0.01  # 스텝당 패널티
    
    # 보상 가중치
    RETURN_REWARD_SCALE = 100  # 수익률 보상 스케일
    TRADE_ACTIVITY_REWARD = 0.05  # 적절한 거래 활동 보상
    EXCESSIVE_TRADE_PENALTY = 0.1  # 과도한 거래 패널티
    COMMISSION_PENALTY_SCALE = 50  # 수수료 패널티 스케일
    
    # 탐색 설정
    INITIAL_RANDOM_EPISODES = 50  # 초기 랜덤 액션 에피소드
    RANDOM_ACTION_PROB = 0.2  # 초기 랜덤 액션 확률
    ACTION_NOISE_STD = 0.1  # 액션 노이즈 표준편차
    
    # 환경 보상 타입
    REWARD_TYPE = "trade_encouraging"
else:
    # 기본 설정 유지
    REWARD_TYPE = "original" 
"""
로컬 LLM 커넥터 모듈
로컬에서 실행되는 LLM 모델을 사용한 텍스트 생성 기능 제공
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import torch

# 하드코딩된 설정값
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 전역 변수로 모델과 토크나이저 캐싱
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def load_local_llm():
    """
    로컬 LLM 모델을 로드합니다.
    4-bit 양자화를 적용하여 GPU 메모리를 최적화합니다.
    모델과 토크나이저는 전역 변수에 캐싱되어 재사용됩니다.
    
    Returns:
        (model, tokenizer) 튜플
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print(f"🔄 로컬 LLM 모델 로딩 중: {MODEL_NAME}")
    print(f"📍 디바이스: {DEVICE}")
    
    try:
        # 토크나이저 로드
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 4-bit 양자화 설정
        if DEVICE == "cuda":
            # GPU에서 4-bit 양자화 사용
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            # CPU에서는 양자화 없이 로드 (메모리 제약 고려)
            print("⚠️ CPU 모드: 양자화 없이 로드합니다. 메모리 사용량이 클 수 있습니다.")
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            _model = _model.to(DEVICE)
        
        print("✅ 로컬 LLM 모델 로드 완료")
        return _model, _tokenizer
    
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        raise


def generate_response(prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
    """
    로컬 LLM을 사용하여 프롬프트에 대한 텍스트 응답을 생성합니다.
    
    Args:
        prompt: 입력 프롬프트
        max_length: 생성할 최대 토큰 길이 (기본값: 512)
        temperature: 생성 온도 (기본값: 0.7, 높을수록 더 창의적)
    
    Returns:
        생성된 텍스트 응답
    """
    # 모델과 토크나이저 로드
    model, tokenizer = load_local_llm()
    
    # 프롬프트 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 생성 파라미터 설정
    generation_config = {
        "max_length": max_length,
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    try:
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거 (생성된 부분만 반환)
        if prompt in generated_text:
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        
        return response
    
    except Exception as e:
        raise RuntimeError(f"텍스트 생성 중 오류 발생: {str(e)}")


def is_gpu_available() -> bool:
    """
    GPU 사용 가능 여부를 확인합니다.
    
    Returns:
        GPU 사용 가능 여부
    """
    return torch.cuda.is_available()


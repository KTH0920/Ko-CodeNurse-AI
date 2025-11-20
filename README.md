# Ko-CodeNurse-AI
한국어 기반의 리서치, 코딩, 간호 지원 기능을 통합한 AI

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```bash
# FastAPI API URL (Streamlit 프론트엔드에서 사용)
FASTAPI_API_URL=http://localhost:8000

# OpenAI API 키 (FastAPI 백엔드에서 사용)
OPENAI_API_KEY=your_openai_api_key_here

# FastAPI 서버 설정 (선택적)
# HOST=0.0.0.0
# PORT=8000
```

### 환경 변수 설명

- `FASTAPI_API_URL`: Streamlit 앱이 FastAPI 백엔드를 호출할 때 사용하는 URL
  - 기본값: `http://localhost:8000`
  - 프로덕션 환경에서는 실제 서버 URL로 변경
- `OPENAI_API_KEY`: OpenAI API 키 (LLM 답변 생성에 필요)
- `HOST`: FastAPI 서버 호스트 (기본값: `0.0.0.0`)
- `PORT`: FastAPI 서버 포트 (기본값: `8000`)
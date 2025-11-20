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

## Docker를 사용한 배포

### FastAPI 백엔드 빌드 및 실행

```bash
# Docker 이미지 빌드
docker build -f Dockerfile.backend -t ko-codenurse-backend .

# Docker 컨테이너 실행
docker run -d \
  --name ko-codenurse-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -v $(pwd)/vector_store:/app/vector_store \
  ko-codenurse-backend
```

### 환경 변수 설정 (Docker)

Docker 실행 시 환경 변수를 설정할 수 있습니다:

```bash
docker run -d \
  --name ko-codenurse-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -v $(pwd)/vector_store:/app/vector_store \
  ko-codenurse-backend
```

### 벡터 저장소 마운트

벡터 저장소는 볼륨으로 마운트하여 데이터를 영구 보존할 수 있습니다:

```bash
-v $(pwd)/vector_store:/app/vector_store
```
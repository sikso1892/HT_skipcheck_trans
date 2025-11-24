# NAC/HT LLM 처리 파이프라인

## 코드 구성
- 흐름: `main.py` (엑셀 읽기 → 프롬프트 생성 → OpenAI 호출 → 엑셀 저장), `build_prompt.py` (프롬프트 텍스트 정의).
- 간략 다이어그램:
  ```
  Excel 입력
      ↓ 읽기 (main.py)
  오류 유형 고르고 번역할 행 필터/선택
      ↓ 프롬프트 생성 (build_prompt.py)
  OpenAI 호출 (skipcheck → translation)
      ↓ 결과 기록
  Excel 출력 저장 (중간 체크포인트 포함)
  ```

## 실행 절차
1. `OPENAI_API_KEY`를 환경변수 또는 `.env`로 설정.
2. 패키지 설치: requirements.txt
3. `main.py`의 `INPUT_PATH` / `OUTPUT_PATH`를 대상 파일 경로로 수정.
4. 실행: `python main.py` >> CLI보단 그냥 F5 눌러서 실행

## 입력/출력 포맷
- 필수 컬럼: `status`, `Translation`(또는 `Translation1/Translation2`), `Src language`, `Origin`, `Check boxes`, `Tgt language`.
- 처리 대상: `status`가 `In progress OR No participation`이고 번역 칸이 비어 있는 행.
- 체크포인트: 30건 처리할 때마다 중간 저장 파일 생성.

## 에러 및 로그
- 실행 시 같은 디렉터리에 `log_YYYYMMDD_HHMMSS.txt` 생성.
- 응답 실패로 `__ERROR__`가 기록된 행은 건너뛰므로, 필요 시 해당 행들을 필터링해 다시 실행하거나 재처리.
- 중간 Open AI 서버 장애로 끊길 경우 Check point 파일부터 재실행

## 확장 가이드
- 모델/타임아웃 변경: `skipcheck_gpt` / `trans_gpt` 호출부와 상단의 기본값을 조정. 
- 빠른 검증: 작은 샘플 입력 엑셀을 별도로 준비해 실행해보면 설정 변경 후 확인이 빠름.

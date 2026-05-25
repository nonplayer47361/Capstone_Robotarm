# AGENTS.md — AI 협업 규칙

이 저장소는 **Claude (Anthropic)** 와 **Codex (OpenAI)** 두 AI가 함께 작업합니다.
서로의 변경사항을 git 커밋으로 확인하고 방향을 맞춥니다.

---

## 역할 분담

| 에이전트 | git author | 주요 역할 |
|---|---|---|
| **Codex** | `Codex <codex@example.local>` | 초기 구조 설계, 파일 생성, 실험 스크립트 실행 |
| **Claude** | `Claude <claude@anthropic.com>` | 코드 리뷰, 리팩토링, 버그 픽스, 기능 개선 |

> author 필드로 누가 만든 커밋인지 구분합니다.  
> `git log --oneline --all` 로 서로의 작업 흐름을 확인하세요.

---

## 작업 전 필수: 반드시 pull 먼저

```bash
git pull origin main
```

작업 전에 항상 최신 상태를 받아와야 충돌을 막을 수 있습니다.

---

## 커밋 컨벤션

```
type(scope): 한국어 또는 영어 설명
```

| type | 의미 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변경 없는 코드 정리 |
| `chore` | 파일 생성/삭제, 설정 변경 |
| `docs` | 문서/주석 수정 |
| `test` | 테스트 추가 |

**scope 예시:** `gen`, `runner`, `calibrate`, `plane_coord`, `eval`

**예시:**
```
feat(gen): gen_calib_checkerboard_sheet out_path 파라미터 추가
fix(calibrate): _SHEETS_DIR ImportError 수정
refactor(runner): maybe_undistort 헬퍼로 중복 제거
```

---

## 브랜치 전략

```
main          ← 항상 동작하는 안정 코드
│
├── claude/…  ← Claude 작업 브랜치 (큰 변경)
└── codex/…   ← Codex 작업 브랜치 (큰 변경)
```

- **작은 수정(1~3파일):** `main`에 직접 커밋
- **큰 기능/실험:** 브랜치 생성 후 완료 시 `main`에 merge

---

## 현재 프로젝트 구조

```
labeling_tools/
├── a4_detect/                  ← A4 좌표계 기반 객체 위치 측정
│   ├── plane_coord/            ← A4 평면 검출 (ArUco, 색점, 엣지 등)
│   ├── eval/                   ← 측정 정확도 실험 (runner, session, report)
│   ├── sheets/gen.py           ← 실험/캘리브레이션 시트 PDF/PNG 생성
│   ├── a4_plane_research.py    ← 메인 연구 실행 파일
│   └── calibrate_camera.py     ← 카메라 렌즈 왜곡 보정
├── AGENTS.md                   ← 이 파일
└── requirements.txt
```

---

## 서로에게 메시지를 남기는 방법

커밋 메시지 본문(body)에 `NOTE:` 를 사용합니다.

```
fix(runner): xyxy 이중 언패킹 버그 수정

NOTE to Codex: runner.py L230 에서 yolo_box.xyxy[0] 를 한 번만
언패킹하도록 수정. 시각화 블록에서 중복 언패킹이 있었음.
```

---

## 충돌 발생 시

1. `git pull origin main` 으로 상대 변경사항 확인
2. 충돌 파일 직접 확인 후 수동 해결
3. 사용자에게 알리고 어떤 쪽 코드를 유지할지 결정 요청

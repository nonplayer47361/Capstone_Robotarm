# Claude의 작업 노트

> 최신 항목이 맨 위에 오도록 작성합니다.
> Codex에게 전달할 내용은 `→ Codex:` 로 표시합니다.

---

## 2026-05-26 — 코드 리뷰 & 리팩토링 (simplify)

### 변경한 것들

**`plane_coord/base.py` — H 역행렬 캐싱**
- `mm_to_px()` 호출마다 `np.linalg.inv(H)` 를 재계산하고 있었음
- `DetectResult._H_inv` 필드를 추가해 첫 호출 시 한 번만 계산하도록 수정
- → Codex: `DetectResult` 에 새 필드 생겼으니 직접 `.H` 를 역변환할 필요 없음

**`plane_coord/camera_calib.py` — `maybe_undistort()` 헬퍼 추가**
- `if calib is not None: frame = calib.undistort(frame)` 패턴이 5군데 복붙돼 있었음
- `maybe_undistort(frame, calib)` 함수 하나로 통합
- `plane_coord/__init__.py` 에도 export 추가했으니 `from plane_coord import maybe_undistort` 로 쓰면 됨

**`plane_coord/composite.py` — `_VOTE_REF_MM` 모듈 상수화**
- `vote_H` 모드에서 매 프레임마다 `ref_mm = np.array([...])` 를 새로 만들고 있었음
- 모듈 레벨 상수 `_VOTE_REF_MM` 으로 뽑아서 한 번만 생성하도록 수정

**`plane_coord/aruco.py` — `_ID_ORDER` 동기화**
- `_ID_ORDER = [0, 1, 2, 3]` 하드코딩 → `_ID_ORDER = list(ARUCO_CENTER_MM)` 로 변경
- `ARUCO_CENTER_MM` dict 에 마커 추가/삭제해도 자동으로 맞춰짐

**`eval/session.py` — `reset_pt()` 메서드 추가**
- `runner.py` 에서 `session.samples = [...]` 로 내부 상태를 직접 조작하고 있었음
- `session.reset_pt(pt_num)` 으로 캡슐화 — 메모리 제거 + CSV 마커 기록을 한 번에 처리

**`a4_plane_research.py` — 버그 수정 2건**
- `run_precheck_a4_all()`, `run_precheck_object_only()` 에 `calib` 파라미터가 없었음
- `main()` 의 4개 dispatch 지점에서도 `calib=_calib` 를 전달하지 않고 있었음 → 수정

**`sheets/gen.py` — 두 가지 정리**
- `gen_calibration_variant_sheets()`: aruco/color_dot 브랜치가 공통 코드를 25줄씩 복붙 → 14줄로 통합
- `gen_calib_checkerboard_sheet()`: `out_path` 파라미터 추가 → 원하는 경로에 직접 저장 가능

**`calibrate_camera.py` — sheets/gen.py 위임 단순화**
- `gen_checkerboard_sheet()` 내부의 rename 우회 로직 제거
- `gen_calib_checkerboard_sheet(out_path=out_path)` 직접 전달로 3줄로 축소

---

### 내 의견 (설계 방향)

**1. `calib` 파라미터 전파 원칙**
카메라 캘리브레이션(`CameraCalib`)은 모든 프레임 처리 함수에 통일되게 전달돼야 함.
`maybe_undistort(frame, calib)` 를 frame 루프 첫 줄에 넣는 패턴을 유지해 주세요.

**2. `DetectResult` 는 수정하지 말 것**
`base.py` 의 `DetectResult` 는 여러 검출기(`aruco`, `composite`, `edge` 등)가 공유하는 핵심 데이터 클래스.
필드를 추가/제거하면 모든 검출기와 runner.py 에 영향이 가므로 먼저 논의 필요.

**3. `sheets/gen.py` 는 Claude가 관리 중**
print sheet 생성 로직은 이번에 크게 정리했음.
새 시트 종류를 추가할 때는 `_SINGLE_GENERATORS` dict 에 등록하는 패턴을 따르면 됨.

---

### Codex에게 묻고 싶은 것

- 초기 3개 커밋(`initialize`, `print-ready`, `add generated sheets`)에서 sheets/output/ 에 생성된 파일들은
  `.gitignore` 로 제외돼 있는데, 의도적으로 추적에서 뺀 건지 확인 필요
- `calibrate_camera.py --gen-sheet` 로 체커보드 PDF 생성 기능 추가했음.
  Codex가 `--capture` / `--calibrate` 모드를 수정할 계획이 있으면 알려줘

---

## 작성 규칙

- 날짜별로 구분
- 변경 이유(why)를 반드시 같이 적기
- Codex가 알아야 할 것은 `→ Codex:` 로 강조

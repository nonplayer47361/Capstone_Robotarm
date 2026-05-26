# A4 + YOLO 오늘 실험 절차

이 문서는 4장 출력물 기준으로 실제 종이 위 좌표 추정 실험을 진행하기 위한 실행 순서다.
모든 명령은 `a4_detect/` 폴더에서 실행한다.

```powershell
cd "C:\Users\dhtmd\OneDrive\바탕 화면\robotarm\yolo\labeling_tools\a4_detect"
```

## 핵심 원칙

- 체커보드는 카메라 왜곡 보정 전용으로만 사용한다.
- 좌표계 방식 비교는 `edge`, `aruco`, `grid` 3개를 기준으로 한다.
- 수평/기울기 실험은 반드시 `--condition`으로 분리한다.
  - 예: `level`, `tilt_low`, `tilt_mid`, `tilt_high`
- `tilt_score`는 실제 각도(degree)가 아니라 이미지에서 A4가 얼마나 원근 왜곡되어 보이는지 나타내는 지표다.
  - 1.0에 가까울수록 수직 촬영에 가까움
  - 낮을수록 기울어진 프레임
- 약통뚜껑 모델의 클래스명이 `pill_cap`이면, 동전/페트병뚜껑/돌멩이를 같은 모델로 테스트할 때는 `--expected-class pill_cap`을 같이 사용한다.

## 추천 실행: 배치 메뉴

```powershell
.\RUN_TODAY_EXPERIMENT.bat
```

메뉴에서 다음 순서대로 진행한다.

1. `1` Camera calibration capture
2. `2` Camera calibration preview
3. `3` A4 methods precheck
4. `4` YOLO object-only precheck
5. `5` A4 + YOLO integration precheck
6. `7` edge -> aruco -> grid 좌표 측정
7. `8` CSV 리포트 재생성

## STEP 1: 카메라 왜곡 보정

```powershell
python calibrate_camera.py --capture
python calibrate_camera.py --preview --calib calib_camera0.json
```

체커보드를 여러 각도와 거리에서 20장 이상 촬영한다.
RMS가 1.0px 미만이면 우선 사용 가능하고, 1.5px 이상이면 다시 촬영한다.

## STEP 2: A4 검출 방식 선행 테스트

수평 조건:

```powershell
python a4_plane_research.py --precheck --precheck-target a4 --all-methods --condition level --calib calib_camera0.json
```

기울기 조건:

```powershell
python a4_plane_research.py --precheck --precheck-target a4 --all-methods --condition tilt_low --calib calib_camera0.json
```

성공률 80% 이상이면 해당 조건에서 GO로 본다.

## STEP 3: YOLO 객체 단독 확인

종이 위가 아닌 환경에서도 약통뚜껑 탐지가 되는지 확인한다.

```powershell
python a4_plane_research.py --precheck --precheck-target object --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --condition level --calib calib_camera0.json
```

동전/페트병뚜껑/돌멩이를 같은 모델로 일반화 테스트할 때도 먼저 단독 확인한다.

```powershell
python a4_plane_research.py --precheck --precheck-target object --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type coin --condition level --calib calib_camera0.json
python a4_plane_research.py --precheck --precheck-target object --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type bottle_cap --condition level --calib calib_camera0.json
python a4_plane_research.py --precheck --precheck-target object --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type stone --condition level --calib calib_camera0.json
```

## STEP 4: A4 + YOLO 통합 확인

```powershell
python a4_plane_research.py --precheck --precheck-target both --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --condition level --calib calib_camera0.json
```

`edge`, `aruco`, `grid`를 각각 확인한다.

```powershell
python a4_plane_research.py --precheck --precheck-target both --method edge  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --condition level --calib calib_camera0.json
python a4_plane_research.py --precheck --precheck-target both --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --condition level --calib calib_camera0.json
python a4_plane_research.py --precheck --precheck-target both --method grid  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --condition level --calib calib_camera0.json
```

## STEP 5: 좌표 오차 측정

약통뚜껑, 수평 조건:

```powershell
python a4_plane_research.py --eval --method edge  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
python a4_plane_research.py --eval --method grid  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
```

약통뚜껑, 기울기 조건:

```powershell
python a4_plane_research.py --eval --method edge  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition tilt_low --calib calib_camera0.json
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition tilt_low --calib calib_camera0.json
python a4_plane_research.py --eval --method grid  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --one-point --manual --repeats 5 --condition tilt_low --calib calib_camera0.json
```

동전/페트병뚜껑/돌멩이를 약통뚜껑 모델로 테스트하는 경우:

```powershell
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type coin       --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type bottle_cap --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type stone      --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json
```

## STEP 6: 리포트 비교

```powershell
python a4_plane_research.py --report --csv "eval_logs\eval_pill_cap_aruco_level_YYYYMMDD_HHMMSS.csv"
```

비교 기준은 다음 순서로 본다.

1. A4 검출 성공률
2. YOLO 탐지 성공률
3. 좌표 평균 오차
4. 좌표 p90 오차
5. `tilt_score` 범위
6. 5mm / 10mm / 15mm 이내 비율

## 오늘 목표 체크리스트

- [ ] 체커보드로 `calib_camera0.json` 생성
- [ ] 수평 조건 A4 방식 선행 테스트
- [ ] 기울기 조건 A4 방식 선행 테스트
- [ ] 약통뚜껑 YOLO 단독 테스트
- [ ] 약통뚜껑 A4+YOLO 통합 테스트
- [ ] 약통뚜껑 `edge/aruco/grid` 좌표 측정
- [ ] 동전 크기 변화 테스트
- [ ] 페트병뚜껑 크기 변화 테스트
- [ ] 돌멩이 불규칙 형태 테스트
- [ ] 리포트 비교 후 1차 기준 방식 선정

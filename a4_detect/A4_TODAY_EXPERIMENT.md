# A4 + YOLO Pill-Cap Experiment

이 문서는 약통뚜껑(`pill_cap`) 하나만 사용해서 A4 좌표계 검출과 YOLO 객체 탐지를 통합 검증하는 절차다.
동전, 페트병뚜껑, 돌멩이 테스트는 이번 실험 범위에서 제외한다.

모든 명령은 `a4_detect/` 폴더에서 실행한다.

```powershell
cd "C:\Users\dhtmd\OneDrive\바탕 화면\robotarm\yolo\labeling_tools\a4_detect"
```

## 사용 출력물

- 체커보드: 카메라 왜곡 보정 전용
- 001 edge 중앙점 시트: A4 외곽선 방식 테스트
- 006 aruco 중앙점 시트: ArUco 방식 테스트
- 016 grid 중앙점 시트: 그리드 방식 테스트

중앙 가이드 원은 약통뚜껑 배치를 돕기 위한 참고 표시다. 사람이 직접 놓기 때문에 완전한 중심 일치는 요구하지 않고, 좌표 오차로 기록한다.

## 권장 실행

```powershell
.\RUN_TODAY_EXPERIMENT.bat
```

배치 메뉴에서 아래 순서로 진행한다.

1. `1` Camera calibration capture
2. `2` Camera calibration preview
3. `3` A4 methods precheck
4. `4` YOLO object-only precheck
5. `5` A4 + YOLO integration precheck
6. `7` edge -> aruco -> grid coordinate evaluation
7. `8` CSV report rebuild

## STEP 1: 카메라 왜곡 보정

체커보드 시트를 카메라 앞에 놓고 다양한 거리와 각도로 20장 이상 촬영한다.

```powershell
python calibrate_camera.py --capture --camera 1
python calibrate_camera.py --preview --calib calib_camera0.json --camera 1
```

기준:

- RMS < 1.0px: 우선 사용 가능
- RMS 1.0~1.5px: 사용 가능하지만 추가 촬영 권장
- RMS >= 1.5px: 다시 촬영 권장

## STEP 2: A4 검출 방식 선행 테스트

수평 조건:

```powershell
python a4_plane_research.py --precheck --precheck-target a4 --all-methods --condition level --calib calib_camera0.json --camera 1
```

기울기 조건:

```powershell
python a4_plane_research.py --precheck --precheck-target a4 --all-methods --condition tilt_low --calib calib_camera0.json --camera 1
```

`edge`, `aruco`, `grid`의 성공률과 재투영 오차를 비교한다. 성공률 80% 이상이면 해당 조건에서 GO로 본다.

## STEP 3: YOLO 객체 단독 확인

A4 없이 약통뚜껑만 카메라에 보여서 탐지가 안정적인지 먼저 확인한다.

```powershell
python a4_plane_research.py --precheck --precheck-target object --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --condition level --calib calib_camera0.json --camera 1
```

그 다음 A4 위에 약통뚜껑을 올려 같은 테스트를 반복해도 된다. 배경이 바뀌었을 때 탐지율이 떨어지는지 확인하는 용도다.

## STEP 4: A4 + YOLO 통합 확인

각 방식별로 A4 좌표계 검출과 약통뚜껑 탐지가 동시에 되는지 확인한다.

```powershell
python a4_plane_research.py --precheck --precheck-target both --method edge  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --condition level --calib calib_camera0.json --camera 1
python a4_plane_research.py --precheck --precheck-target both --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --condition level --calib calib_camera0.json --camera 1
python a4_plane_research.py --precheck --precheck-target both --method grid  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --condition level --calib calib_camera0.json --camera 1
```

## STEP 5: 좌표 오차 측정

수평 조건에서 3개 방식을 각각 측정한다.

```powershell
python a4_plane_research.py --eval --method edge  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json --camera 1
python a4_plane_research.py --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json --camera 1
python a4_plane_research.py --eval --method grid  --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --one-point --manual --repeats 5 --condition level --calib calib_camera0.json --camera 1
```

기울기 조건도 동일하게 진행하되 `--condition tilt_low`, `tilt_mid`, `tilt_high`처럼 조건명을 분리해서 저장한다.

## STEP 6: 리포트 비교

생성된 CSV를 기준으로 리포트를 다시 만들 수 있다.

```powershell
python a4_plane_research.py --report --csv "eval_logs\eval_pill_cap_aruco_level_YYYYMMDD_HHMMSS.csv"
```

비교 기준:

1. A4 검출 성공률
2. YOLO 탐지 성공률
3. 평균 좌표 오차
4. p90 좌표 오차
5. 5mm / 10mm / 15mm 이내 비율
6. 수평 조건과 기울기 조건의 차이

## 오늘 체크리스트

- [ ] 체커보드로 `calib_camera0.json` 생성
- [ ] 수평 조건 A4 방식 선행 테스트
- [ ] 기울기 조건 A4 방식 선행 테스트
- [ ] 약통뚜껑 YOLO 단독 테스트
- [ ] 약통뚜껑 A4+YOLO 통합 테스트
- [ ] 약통뚜껑 `edge/aruco/grid` 수평 좌표 측정
- [ ] 약통뚜껑 `edge/aruco/grid` 기울기 좌표 측정
- [ ] 리포트 비교 후 1차 기준 방식 선정

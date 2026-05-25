# Full Auto YOLO Expansion

이 폴더는 기존 검수 기반 연구 파이프라인과 분리된 풀오토 실험 공간입니다.

목표는 초기 수동 seed 라벨을 출발점으로 삼아, 모델이 높은 신뢰도로 검출한 라벨만 자동 채택하고 재학습하면서 라벨 수가 어디까지 늘어나는지 측정하는 것입니다.

## 핵심 원칙

```text
기존 research_runs는 원본 seed/holdout 보관용으로만 사용
풀오토 결과는 풀오토/runs/<target>/ 아래에만 저장
사람 검수 없이 자동 채택 기준을 통과한 라벨만 학습 데이터에 누적
각 반복마다 신규 채택 수, 누적 라벨 수, 미채택 수를 리포트로 저장
```

## 실행 전 조건

먼저 기존 파이프라인에서 초기 seed 50장 라벨링이 완료되어 있어야 합니다.

```text
research_runs/blueberry/01_manual_seed_dataset
research_runs/strawberry/01_manual_seed_dataset
```

holdout 라벨링은 필수는 아니지만, 최종 성능 검증까지 하려면 기존 파이프라인에서 holdout 50장 라벨링도 해두는 것이 좋습니다.

## 실행

```bat
cd /d "C:\Users\dhtmd\OneDrive\바탕 화면\robotarm\yolo\labeling_tools\풀오토"
RUN_FULL_AUTO.bat
```

또는 직접 실행:

```bat
python full_auto_pipeline.py --config configs\blueberry.json
python full_auto_pipeline.py --config configs\strawberry.json
```

상태만 확인:

```bat
python full_auto_pipeline.py --config configs\blueberry.json --status
```

## 출력 구조

```text
풀오토/runs/<target>/
  datasets/
    iter_00_seed/
    iter_01_auto/
    iter_02_auto/
    ...
  models/
    iter_00_seed_model/
    iter_01_auto_model/
    ...
    final_model/
  reports/
    full_auto_summary.json
    full_auto_summary.md
    iter_01_accepted.txt
    iter_01_rejected.txt
```

## 해석 기준

좋은 결과:

```text
반복이 진행될수록 누적 라벨 수가 증가
후반으로 갈수록 신규 채택 수가 줄어듦
최종적으로 전체 source 이미지 중 높은 비율이 자동 라벨링됨
```

주의할 결과:

```text
초기부터 신규 채택 수가 거의 없음
너무 낮은 기준으로 많은 라벨이 한 번에 채택됨
객체가 아닌 배경/그림자를 높은 confidence로 채택함
```

이 실험의 핵심 지표는 최종 정확도만이 아니라, `수동 seed 50장으로부터 자동 라벨이 얼마나 확장되는가`입니다.

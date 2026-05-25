# Review Minimization Strategy

이 문서는 초기 수동 라벨 50장 이후 사람 검수량을 줄이기 위한 현재 전략을 정리한다.

## 핵심 원칙

초기 seed 50장과 holdout 50장은 사람이 직접 라벨링하거나 전수 검수한다. 이 두 구간은 모델이 아직 불안정하거나 최종 평가 기준이 되기 때문이다.

Stage 1 자동 라벨은 사람이 검수한다. Stage 1 이후부터는 이미 한 번 검수된 모델을 사용하므로, 고신뢰 자동 라벨은 자동 승인하고 불확실한 라벨만 검수한다.

## 현재 적용 방식

Stage 2/3 자동 라벨링 후 각 라벨을 다음 기준으로 평가한다.

```text
confidence >= auto_accept_confidence
탐지 박스 수 == 1
박스 면적 비율이 설정 범위 안에 있음
박스 가로세로 비율이 설정 범위 안에 있음
선택 사항: flip consistency IoU가 설정값 이상
```

통과한 라벨은 `review_status.json`에 미리 reviewed로 표시된다. 검수 도구는 `--unchecked-only`로 열리므로 자동 승인된 라벨은 숨겨지고, 사람이 볼 필요가 있는 라벨만 표시된다.

## 감사 샘플

자동 승인된 라벨도 일부는 사람이 확인해야 한다. 이를 위해 `auto_accept_audit_ratio`를 사용한다.

예시:

```json
"auto_accept_audit_ratio": 0.05
```

자동 승인 대상 중 약 5%를 deterministic sample로 남겨 검수 창에 표시한다. 매번 같은 이미지가 선택되므로 실험 재현성이 유지된다.

## Config 예시

블루베리는 구형에 가까우므로 aspect ratio를 비교적 좁게 둔다.

```json
"auto_accept_confidence": 0.85,
"auto_accept_min_area_ratio": 0.0003,
"auto_accept_max_area_ratio": 0.35,
"auto_accept_min_aspect_ratio": 0.55,
"auto_accept_max_aspect_ratio": 1.75,
"auto_accept_require_single_detection": true,
"auto_accept_audit_ratio": 0.05,
"auto_accept_consistency_iou": 0.0
```

딸기는 형태 변화가 더 크므로 aspect ratio를 더 넓게 둔다.

```json
"auto_accept_min_aspect_ratio": 0.45,
"auto_accept_max_aspect_ratio": 2.50
```

`auto_accept_consistency_iou`를 0보다 크게 설정하면 좌우 반전 예측도 비교한다. 더 안전하지만 추론 시간이 늘어난다.

## 출력 파일

각 자동 라벨 데이터셋에는 다음 파일이 생긴다.

```text
review_status.json
auto_accept_report.json
```

`auto_accept_report.json`에는 자동 승인 수, 검수 필요 수, 감사 샘플 수, 탈락 이유가 기록된다.

## 권장 실행 순서

```text
1. Action 8: holdout 50장 라벨링
2. Action 1: seed 50장 라벨링 후 N
3. Action 11: Stage 1 학습, 자동 라벨, 사람 검수, 재학습
4. Action 5: Stage 2부터 재개
   - review windows를 열면 자동 승인 라벨은 숨겨지고 불확실 라벨만 표시
   - N을 선택하면 Stage 2/3 검수를 완전히 생략
```

## 해석

이 방식은 confidence threshold만 쓰는 방법보다 안전하다. 높은 confidence라도 박스가 너무 작거나 크거나, 형태가 대상과 맞지 않거나, 여러 박스가 동시에 잡히면 사람이 보도록 남긴다.

연구 보고서에서는 이 전략을 "quality-gated auto-acceptance with audit sampling"으로 설명할 수 있다.

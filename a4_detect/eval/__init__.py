"""
eval — A4 좌표 오차 측정 실험 모듈

핵심 지표: YOLO bbox 중심점 → A4 mm 좌표 변환 오차 (mm)

실험 흐름:
  1. 실험 시트(sheet_eval_30pt.png) 출력 → 카메라 아래 고정
  2. A4 평면 검출 (ArUco / composite)
  3. 각 테스트 포인트에 객체(딸기/블루베리) 올려놓기
  4. YOLO bbox 중심점 → px_to_mm() → 예측 mm 좌표
  5. 예측 vs 실제 오차 기록
  6. 리포트 생성: 평균/중앙값/p90/최대 오차, 5mm·10mm 이내 비율 등

핵심 평가 지표:
  - 좌표 평균 오차 (mm)
  - 좌표 중앙값 오차 (mm)
  - 좌표 p90 오차 (mm)
  - 최대 오차 (mm)
  - X / Y 방향 평균 bias (mm)
  - 5mm 이내 성공률 (%)
  - 10mm 이내 성공률 (%)
  - YOLO 탐지 성공률 (%)
  - A4 검출 성공률 (%)

좌표계:
  원점(0,0) = A4 왼쪽 위
  X → 오른쪽  (0~210 mm)
  Y ↓ 아래    (0~297 mm)
"""
from .session import EvalSession, Sample
from .report  import compute_report, print_report, save_report_json

# ── 30점 실험 좌표 격자 ────────────────────────────────────────────────────────
# X: 55, 80, 105, 130, 155 mm (5열)
# Y: 65, 100, 135, 170, 210, 245 mm (6행)
# → 번호 1~30: 왼→오, 위→아래 순서
#
# ※ ArUco 마커(표준 inset25, 20mm) 모서리 좌표:
#     TL(35,35)  TR(175,35)  BL(35,262)  BR(175,262)
# 40mm 캡(반경 20mm) 기준 최소 이격 20mm 필요.
# 격자 코너 (55,65)↔TL(35,35) = 36mm,  (55,245)↔BL(35,262) = 26mm  — 모두 안전.
EVAL_X_MM = [55.0, 80.0, 105.0, 130.0, 155.0]
EVAL_Y_MM = [65.0, 100.0, 135.0, 170.0, 210.0, 245.0]

EVAL_TEST_PTS: list[tuple[int, float, float]] = [
    (num, x, y)
    for num, (y, x) in enumerate(
        ((y, x) for y in EVAL_Y_MM for x in EVAL_X_MM),
        start=1,
    )
]
# 결과: [(1,55,65),(2,80,65),...,(30,155,245)]

__all__ = [
    "EvalSession", "Sample",
    "compute_report", "print_report", "save_report_json",
    "EVAL_TEST_PTS", "EVAL_X_MM", "EVAL_Y_MM",
]

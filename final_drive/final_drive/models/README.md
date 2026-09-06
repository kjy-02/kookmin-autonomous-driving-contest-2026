# 신호등 외함 검출 모델

`traffic_housing_v2_best.onnx`는 대회 통합 런치에서 사용하는 단일 클래스 `traffic_light` 검출 모델입니다. 외함 위치를 찾은 뒤 [`housing_classifier.py`](../final_drive_vision/housing_classifier.py)가 고정 R/Y/L/G 슬롯의 밝기로 신호 상태를 분류합니다.

| 항목 | 명세 |
| --- | --- |
| 모델 | YOLOv8n, 외함 1클래스 |
| 입력 | float32 RGB `[1, 3, 640, 640]` |
| 전처리 | `1/255` 정규화, letterbox 채움값 `114` |
| 출력 | float32 `[1, 5, 8400]`: `cx, cy, width, height, score` |
| 내보내기 | ONNX opset 12, 고정 입력, 내장 NMS 없음 |
| 추론 | OpenCV DNN CPU |

SHA-256: `c3c77b0de3ffc891e9f831ca559c2f8d96904204df30f63ac1fedf8a0c81d169`

차량 OpenCV와의 호환성을 위해 legacy TorchScript 내보내기 경로(`dynamo=False`)로 생성된 파일을 사용합니다. 모델을 교체할 때는 차량의 OpenCV에서 로딩·추론을 확인해야 합니다. 위 명세와 해시는 모델 식별 정보이며 정확도 측정 결과는 아닙니다.

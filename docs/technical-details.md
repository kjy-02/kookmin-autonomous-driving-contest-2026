# `final_drive` 기술 상세

## 설계 목표

본선 미션별 알고리즘을 독립적으로 개발하면서도, 실제 모터 명령의 소유권은 하나의 안전 감독기에 모으는 것이 핵심입니다. `camera_drive`와 `cone_drive`는 각자의 후보 명령만 만들고, `drive_supervisor`가 센서 상태·신호등·키보드 제어·현재 모드에 따라 최종 명령을 선택합니다.

## 노드와 토픽

| 노드 | 구독 | 발행 | 역할 |
| --- | --- | --- | --- |
| `final_camera_drive` | `/image_raw`, `/scan`, `/final_drive/mode`, `/final_drive/cone_status`, `/final_drive/overtake_lidar_ignore` | `/final_drive/camera_cmd`, `/final_drive/camera_lane_state`, `/final_drive/overtake_state` | 차선 추종과 차량형 장애물 회피 |
| `final_cone_drive` | `/scan` | `/final_drive/cone_cmd`, `/final_drive/cone_lane_status` | 라바콘 인식과 중앙 경로 추종 |
| `traffic_vision` | `/image_raw` | `/traffic_light_state` | 신호등 외함 검출과 R/Y/L/G 상태 분류 |
| `keyboard_stop` | 터미널 입력 | `/final_drive/manual_stop` | STOP/RUN 명령과 heartbeat |
| `final_drive_supervisor` | 위 후보 명령·상태, `/scan`, `/traffic_light_state`, `/final_drive/manual_stop` | `/xycar_motor`, `/final_drive/mode`, `/final_drive/traffic_state`, `/final_drive/cone_status`, `/final_drive/manual_stop_state` | 모드 전환, 우선순위, timeout, 출력 제한 |

## 1. 카메라 차선 추종

구현: [`camera_drive_node.cpp`](../final_drive/final_drive/src/camera_drive_node.cpp)

1. 입력 영상을 640×480으로 맞추고 하단 `y=300..379` ROI만 사용합니다.
2. 회색조 → 5×5 Gaussian blur → Canny(60, 80) → `HoughLinesP` 순으로 선분을 구합니다.
3. 기울기 절댓값 0.15 이하의 수평 성분을 제거합니다.
4. 선분 주변의 명도와 채널 차이를 이용해 흰색 차선 점수를 계산합니다.
5. 좌·우 후보 중 이전 위치에 가까우면서 흰색 점수가 높은 선분들을 합쳐 대표선을 만듭니다.
6. 기준 행에서 좌·우 x좌표를 구합니다. 한쪽만 보이면 380 px 기준 차폭으로 반대편을 추정합니다.
7. 차선 중앙 오차에 동적 비례 이득과 미분항을 적용합니다. 오차가 작을 때는 흔들림을 줄이고, 큰 오차에서는 빠르게 조향합니다.
8. 차선 곡률과 최종 조향량에 비례해 직선 속도에서 감속합니다.

차선이 잠시 사라지면 마지막 조향으로 저속 주행하고, 기본 1초 이상 검출되지 않으면 후보 속도를 0으로 만듭니다.

## 2. 신호등 인식

구현: [`traffic_housing_detector.py`](../final_drive/final_drive/final_drive_vision/traffic_housing_detector.py), [`housing_classifier.py`](../final_drive/final_drive/final_drive_vision/housing_classifier.py), [`node.py`](../final_drive/final_drive/final_drive_vision/node.py)

### 외함 검출

- YOLOv8n 단일 클래스 `traffic_light`
- OpenCV DNN CPU 추론
- 고정 입력 `[1, 3, 640, 640]`, RGB, 1/255 정규화, letterbox 값 114
- 출력 `[1, 5, 8400]`, ONNX opset 12, 내장 NMS 없음
- 기본 confidence 0.35
- 기본 영상에서 `x=96..543`, `y=0..239`만 유효한 신호등 영역으로 사용
- 상자 전체가 ROI 안에 있는 후보 중 confidence가 가장 높은 외함 하나 선택

### 4구 상태 분류

검출 상자를 R/Y/L/G 고정 순서로 4등분합니다. 각 슬롯 원 내부의 V 채널에서 밝은 상위 20% 평균과 포화 픽셀 비율을 측정합니다. LED가 카메라에서 흰색에 가깝게 포화되는 특성 때문에 hue보다 위치와 밝기를 우선합니다.

| 값 | 상태 |
| --- | --- |
| 0 | NONE |
| 1 | RED |
| 2 | YELLOW |
| 3 | GREEN |
| 4 | GREEN_LEFT |

통합 런치에서는 비전 노드의 필터를 1프레임으로 두고 감독기가 최종 합의를 담당합니다. RED/YELLOW는 기본 2프레임, GREEN/GREEN_LEFT는 3프레임 연속 확인합니다. 시작 latch는 일반 GREEN만 해제하므로 초기 오인 출발을 막습니다.

## 3. LiDAR 라바콘 주행

구현: [`cone_drive_node.cpp`](../final_drive/final_drive/src/cone_drive_node.cpp)

1. 차량 앞쪽 반평면의 유효 거리점을 극좌표에서 차량 좌표로 변환합니다.
2. 인접 점 사이의 거리로 군집을 만들고, 스캔 배열 경계에서 갈라진 군집은 다시 연결합니다.
3. 군집 지름, PCA 선형성, 현 길이와 sagitta로 추정한 곡률, 센서 쪽을 향한 호인지 여부를 검사합니다.
4. 촘촘한 후보가 이어진 연결 요소는 벽 조각으로 판단하고 몇 스캔 동안 기억해 거친 벽면의 깜박임을 억제합니다.
5. 두 스캔 이상 유지된 후보만 확정 라바콘으로 사용합니다.
6. 약 0.30 m 간격을 기준으로 같은 열을 연결합니다. 한 개가 누락된 0.42~0.72 m 간격도 제한적으로 이어 줍니다.
7. 좌·우 열 폭은 0.70~0.98 m 범위에서 검증합니다.
8. 양쪽 열이 있으면 대응점의 중앙, 한쪽만 있으면 0.38 m 오프셋 경로를 사용합니다.
9. LiDAR가 뒷차축보다 앞에 있다는 점을 반영해 lookahead 조향을 계산합니다.
10. 한쪽 열만 보일 때, S자 조향 반전 중, 라바콘이 차체에 가까울 때 각각 속도 상한을 낮춥니다.

상태 토픽 `/final_drive/cone_lane_status`는 `0=STOP`, `1=한쪽 열 TRACK`, `2=양쪽 열 TRACK`, `3=RECOVERY`입니다.

## 4. 고정 장애물·방해 차량 회피

구현: [`camera_drive_node.cpp`](../final_drive/final_drive/src/camera_drive_node.cpp)

라바콘 주행과 별도로, 카메라 차선 추종 중 LiDAR에서 차량 크기의 전방 군집을 찾습니다.

- 넓은 전방/측면 영역에서 군집을 연결하고 0.48 m보다 긴 군집, 0.50 m보다 긴 연결 요소를 벽으로 제거합니다.
- 이후 전방 1.5 m·좌우 0.40 m 영역만 차량 판단에 사용합니다.
- 한 군집에 5점 이상이 2스캔 연속 나타나야 합니다.
- CAMERA 모드, 라바콘 접근 아님, 실제 양쪽 차선 2프레임 연속, 직선 주행 속도 17 이상일 때만 진입합니다.
- 감독기의 조향 출력이 30° 이상인 곡선에서는 1초간 신규 LiDAR 회피 진입을 차단합니다.
- 왼쪽 장애물이면 오른쪽 차선을, 오른쪽 장애물이면 왼쪽 차선을 선택합니다.
- 통합 런치 기준 `-10` 명령으로 0.5초 제동 후 속도 10으로 회피 주행합니다.
- 원래 장애물 쪽에서 점이 1.5초간 사라지면 정상 추종으로 복귀합니다.

이 로직은 고정된 고장 차량과 느리게 움직이는 방해 차량을 동일한 “소형 차량형 객체”로 다룹니다. 실제 대회 rosbag이 추가되면 두 상황을 분리해 검출 거리·완료 시간을 검증해야 합니다.

## 5. 감독기와 fail-safe

구현: [`drive_supervisor_node.cpp`](../final_drive/final_drive/src/drive_supervisor_node.cpp), [`final_drive.launch.py`](../final_drive/final_drive/launch/final_drive.launch.py)

### 모드 전환

- 기본 모드는 CAMERA입니다.
- 양쪽에 형상 검증된 라바콘이 비슷한 전방 위치에서 보이고 폭·중앙 정렬 조건을 만족하면 접근 상태가 됩니다.
- 감독기는 CAMERA 조향을 유지하면서 속도를 8 이하로 제한합니다.
- 라바콘 노드가 양쪽 열을 안정적으로 추적하면 CONE로 전환합니다.
- 라바콘 노드가 RECOVERY를 마치고 STOP을 내면 CAMERA로 돌아갑니다.
- 기본 1초 재진입 cooldown 뒤 다음 바퀴의 라바콘 구간을 다시 인식할 수 있습니다.

### 정지 우선 조건

| 조건 | 기본값 | 동작 |
| --- | ---: | --- |
| 선택된 주행 명령 timeout | 0.30 s | 즉시 속도 0 |
| LiDAR timeout | 0.30 s | CAMERA 주행 정지 |
| 신호등 토픽 timeout | 0.75 s | 전체 주행 정지 |
| 키보드 heartbeat timeout | 0.50 s | STOP latch, STOP→RUN 재무장 필요 |
| RED/YELLOW 확인 | 2 frames | 전체 주행 정지 latch |
| GREEN/GREEN_LEFT 확인 | 3 frames | 정지 latch 해제 |
| 초기 일반 GREEN 미확인 | 3 frames | 출발 금지 |

정지 명령은 출력 보간보다 우선하므로 감속 램프 때문에 제동이 늦어지지 않습니다. 조향은 30 ms 제어 루프에서 한 번에 최대 12°씩, 정상 양의 속도는 가속 0.60·감속 1.20씩 변화합니다.

## 6. 데이터와 검증 도구

- `bag_reader.py`: ROS 2 sqlite bag의 raw/compressed 이미지 CDR 디코딩
- `bag_frame_viewer.py`: 프레임 탐색, 상태·ROI·검출 상자 시각화
- `prepare_traffic_dataset.py`: 초기 5클래스 YOLO 데이터셋 생성
- `prepare_traffic_housing_dataset.py`: 단일 외함 클래스 v2 데이터셋 생성
- `offline_evaluate.py`: rosbag별 신호 상태 오프라인 평가
- `scripts/lidar_cone_inspector.py`: 라바콘 LiDAR 분석 보조 도구
- `test/`: bag 디코딩, ROI, temporal filter, YOLO 출력 형상, 슬롯 분류, 데이터셋 규칙 테스트

모델 파일과 입출력 명세는 [`models/README.md`](../final_drive/final_drive/models/README.md)에 기록되어 있습니다.

## 알려진 한계

- `GREEN_LEFT` 인식 후 전용 좌회전 경로 계획은 없습니다.
- 랩 카운트, 결승선 검출과 자동 종료는 통합 런치에서 사용하지 않습니다.
- 체크무늬 검출 코드는 보존했지만 실시간 부하를 줄이기 위해 비활성화했습니다.
- 자율주차용 SLAM과 주차 경로 계획은 포함하지 않습니다.
- 제공된 단위 테스트는 Python/OpenCV 환경에서 실행할 수 있지만, 최종 C++ 노드의 검증에는 ROS 2 Humble과 실제/재생 센서 토픽이 필요합니다.

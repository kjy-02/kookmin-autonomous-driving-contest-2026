# final_drive 사용 가이드

카메라 차선 추종, 라이다(LiDAR) 라바콘 주행, 신호등 인식, 차량형 장애물 회피를 통합한 ROS 2 패키지입니다. 알고리즘과 토픽 구조는 [기술 상세](../../docs/technical-details.md), 대회 결과와 시각화는 [프로젝트 소개](../../README.md)에 정리했습니다.

문서에서 사용하는 노드는 개별 기능을 수행하는 프로그램, 토픽은 노드 사이에서 데이터를 주고받는 통신 경로, 런치 파일은 여러 노드와 설정을 한 번에 실행하는 파일입니다.

## 환경과 빌드

- 차량 환경: Xycar Y, Ubuntu 22.04, ROS 2 Humble
- 빌드: C++17, `ament_cmake`, `ament_cmake_python`, OpenCV, `cv_bridge`
- Python: `rclpy`, NumPy, OpenCV
- 장치: 통합 런치는 `usb_cam`과 `xycar_lidar`를 사용합니다. 카메라·신호등 단독 런치에는 `xycar_cam`도 필요합니다.
- 차량의 모터 구동 환경과 카메라·LiDAR 장치 설정은 별도로 준비해야 합니다.

이 디렉터리를 `~/xycar_ws/src/final_drive`에 배치한 뒤 차량에서 빌드합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/xycar_ws
colcon build --symlink-install --packages-select final_drive
source install/setup.bash
```

## 통합 주행

첫 번째 터미널에서 전체 시스템을 실행합니다.

```bash
source ~/xycar_ws/install/setup.bash
ros2 launch final_drive final_drive.launch.py
```

두 번째 대화형 터미널에서 키보드 안전 제어를 실행합니다.

```bash
source ~/xycar_ws/install/setup.bash
ros2 run final_drive keyboard_stop
```

| 키 | 동작 |
| --- | --- |
| `SPACE` | 정지(`STOP`)/주행(`RUN`) 전환 |
| `S` | 정지 |
| `R` | 주행 요청 |
| `Q` | 정지 명령 후 키보드 노드 종료 |

키보드 노드는 정지(`STOP`) 상태로 시작합니다. 초기 정지 신호와 주행(`RUN`) 요청, 최신 센서 입력, 일반 녹색 신호 3프레임 확인이 모두 충족되어야 출발합니다. 키 입력이 전달되도록 키보드 제어 터미널을 선택한 상태로 유지하세요. 주기적으로 보내는 연결 확인 신호가 0.5초 끊기면 정지하며, 다시 정지→주행 순서로 전환해야 합니다.

통합 런치에서는 안전 감독기인 `drive_supervisor`만 최종 `/xycar_motor` 명령을 발행합니다. `camera_drive`는 `/final_drive/camera_cmd`를 만들고, `cone_drive`의 출력 토픽은 `/final_drive/cone_cmd`로 변경해 연결합니다. 감독기는 선택된 주행 명령보다 신호등·키보드의 정지 요청과 입력 갱신 시간 초과에 따른 정지를 우선 적용합니다.

## 최종 파라미터 기준

대회 당일 설정은 [`final_drive.launch.py`](launch/final_drive.launch.py)입니다. 별도 단독 런치와 노드 내부 기본값은 다를 수 있으므로, 최종값은 통합 런치와 해당 노드의 파라미터 처리를 함께 확인합니다. 속도 값은 차량 명령값이며 m/s가 아닙니다.

| 항목 | 통합 실행값 |
| --- | --- |
| 카메라 직선 속도 / 곡선 최소 속도 | `23.0` / `10.0` |
| 라바콘 기본 속도 / 곡선 속도 | `11.0` / `8.5` |
| 라바콘 접근 속도 상한 | `8.0` |
| 추월 제동 / 회피 속도 | `-10.0`으로 0.5초 / `10.0` |
| 추월 진입 최소 속도 | `17.0` |
| 신호 정지 / 재출발 확인 | 빨강·노랑 2프레임 / 녹색 계열 3프레임 |
| 주행 명령 / 라이다 / 신호등 입력 갱신 제한 시간 | `0.30` / `0.30` / `0.75`초 |
| 진단용 화면 | 기본 비활성화 |

벽 분석 영역은 좌우 0.90 m, 차량 검출 영역은 좌우 0.40 m입니다. 전방 범위는 실행 시 둘 다 1.50 m가 적용됩니다. 벽 분석 런치 입력 `1.30`은 코드에서 차량 검출 범위 `1.50` 이상으로 보정됩니다.

## 개별 실행과 진단

| 실행 파일 | 용도와 주의사항 |
| --- | --- |
| `final_drive.launch.py` | 대회용 통합 실행 |
| `camera_traffic_vision.launch.py` | 카메라와 최종 ONNX 신호등 인식만 실행. 모터 명령을 발행하지 않음 |
| `traffic_vision.launch.py` | 기존 영상 토픽을 사용하는 초기 OpenCV 검출기 실행. 기본 모델 경로가 비어 있어 최종 ONNX 설정과 다름 |
| `cone_drive.launch.py` | LiDAR·라바콘 노드만 실행. `/xycar_motor`를 직접 발행하며 감독기의 키보드·신호등 안전 제어를 거치지 않음 |

차량에서 단독 라바콘 실행이나 센서 기록 재생을 할 때는 모터 구동을 분리한 상태에서 확인하세요. 단독 라바콘 런치와 통합 런치를 동시에 실행하면 모터 명령이 충돌할 수 있습니다.

최종 신호등 모델만 확인하려면 다음을 사용합니다.

```bash
ros2 launch final_drive camera_traffic_vision.launch.py
```

통합 실행 중 상태를 확인하는 명령입니다.

```bash
ros2 topic echo /final_drive/mode
ros2 topic echo /final_drive/traffic_state
ros2 topic echo /final_drive/manual_stop_state
```

기록을 재생할 때는 필요한 센서 토픽만 선택하고, 기록된 모터·제어·상태 토픽이 실행 중인 노드와 중복 발행되지 않도록 해야 합니다. 통합 카메라 노드는 `/image_raw`를 사용하므로 압축 영상만 담긴 기록을 그대로 재생하는 것과는 구분해야 합니다.

## 데이터 준비와 오프라인 도구

이 도구들은 개발과 검증을 위해 보존한 것으로, 주행 런치에서는 실행되지 않습니다.

| 도구 | 적용 대상 |
| --- | --- |
| `traffic_housing_dataset_prepare` | 최종 모델용 단일 `traffic_light` 외함 데이터셋 생성 |
| `traffic_housing_color_visualize` | ONNX 검출 상자 또는 정답 표시 상자에 빨강·노랑·좌회전·녹색 네 신호 영역의 분류 결과 표시 |
| `traffic_dataset_prepare` | 초기 5개 범주 신호 상태 분류 실험의 데이터셋 생성 |
| `traffic_bag_viewer` | 프레임 탐색과 초기 OpenCV 신호등·체크무늬 검출 확인 |
| `traffic_bag_evaluate` | 초기 OpenCV 검출기의 상태 평가. 최종 ONNX 모델의 정확도 평가 도구가 아님 |
| `lidar_cone_inspector.py` | LiDAR 라바콘 분석 보조 도구 |

최종 외함 데이터셋을 만드는 예시입니다. `/path/to/...`는 로컬 데이터 경로로 바꿉니다.

```bash
ros2 run final_drive traffic_housing_dataset_prepare \
  /path/to/traffic_rosbag /path/to/traffic_housing_dataset_v2 \
  --driving-background-bag /path/to/rosbag/run_04_cone
```

기본 규칙은 이름에 `_01`, `_02`가 붙은 연속 촬영 기록을 학습용, `_03`이 붙은 기록을 검증용으로 나눕니다. 시간상 인접한 영상이 양쪽에 섞이지 않도록 촬영 기록 단위로 분리합니다.

추출한 이미지에 최종 모델과 네 신호 영역별 상태 분류를 적용하는 예시입니다.

```bash
ros2 run final_drive traffic_housing_color_visualize \
  --images /path/to/traffic_housing_dataset_v2/images/val \
  --model $(ros2 pkg prefix final_drive)/share/final_drive/models/traffic_housing_v2_best.onnx \
  --output /tmp/traffic_slots \
  traffic_red_03_000102 traffic_yellow_03_000090
```

모델 입출력과 해시는 [모델 문서](models/README.md)에 정리했습니다. 체크무늬 검출 코드와 초기 OpenCV 검출기는 개발 이력으로 보존하며, 대회 통합 런치의 신호등 판단에는 최종 외함 모델을 사용합니다.

## 테스트

ROS 2 차량 환경에서 빌드 후 실행합니다.

```bash
cd ~/xycar_ws
colcon test --packages-select final_drive
colcon test-result --verbose
```

Python/OpenCV 기능만 검사하려면 NumPy·OpenCV·pytest가 설치된 환경에서 이 패키지 디렉터리로 이동해 실행합니다.

```bash
python3 -m pytest test -q
```

테스트 범위는 수신 메시지에서 이미지 복원, 관심 영역(ROI) 설정, 신호 필터, YOLO 출력 배열 형태 처리, 네 신호 영역별 상태 분류, 데이터셋 구성 규칙입니다. C++ 노드 빌드와 실제 주행·추론 성능 검증은 별도 ROS 2·차량 환경에서 진행해야 합니다.

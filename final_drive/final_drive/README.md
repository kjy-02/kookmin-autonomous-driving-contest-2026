# final_drive

Single driving package containing camera lane driving, LiDAR cone driving,
traffic-light perception, an archived optional checkerboard detector, and the
final motor supervisor. Hardware
drivers (`xycar_cam` and `xycar_lidar`) remain separate device packages, but no
driving algorithm depends on `cone_drive` or `traffic_vision` anymore. Vehicle
overtaking is integrated into the camera driver and is armed only on a
camera-confirmed straight.

The former `cone_drive` and `traffic_vision` source folders are retained only
as migration backups and contain `COLCON_IGNORE`. Build, launch, and maintain
`final_drive` only.

## Command flow

```text
camera_drive  -> /final_drive/camera_cmd --+
             <- /scan (compact vehicle / long-wall filtering)
                                            +-> drive_supervisor -> /xycar_motor
cone_drive    -> /final_drive/cone_cmd -----+
              -> /final_drive/cone_lane_status (TRACK/RECOVERY/STOP)
/scan         -> drive_supervisor ----------+ (early entrance guard)
/image_raw    -> traffic_vision -> /traffic_light_state
                                  -> drive_supervisor (global stop override)
```

Both driving nodes keep processing their sensors, but neither can publish the
real motor topic directly. `drive_supervisor` is the only `/xycar_motor`
publisher.

## Straight-course vehicle overtaking

The camera driver arms overtaking after both real camera lanes are accepted for
two consecutive image frames. Lane width, lane-center bias, and previous
steering angle are recorded for diagnosis but do not block entry. Detected
lane-center changes are now applied immediately without the former 80 px
two-frame confirmation.
Overtaking is also blocked while the supervisor reports cone approach/CONE
mode, so cone-wall returns cannot arm the maneuver.

Wall classification first uses a wider forward 0.01--1.30 m and lateral
+/-0.90 m LiDAR region. This preserves the full return from a long wall instead
of cropping its short intersection with the vehicle ROI. Returns longer than
the expected 37 x 25 cm target envelope and connected rough wall fragments
longer than 0.50 m are excluded. Only then are points cropped to the forward
0.01--1.00 m and lateral +/-0.40 m vehicle ROI. One compact cluster must contain
at least five points for two consecutive LiDAR scans. A left-side vehicle
selects the right lane boundary and a right-side vehicle selects the left
boundary. The driver requests `-10` active braking for 0.5 seconds, follows the
free lane at the integrated launch speed `10`, and returns after the original vehicle side has remained
clear for 1.5 seconds. Traffic, keyboard, stale-command, and stale-LiDAR stops
remain higher-priority supervisor overrides.

The first eight fields of `/final_drive/overtake_state` remain
`[state,side,left_points,right_points,left_min_m,right_min_m,straight,wall_points]`,
where `state=0/1/2` means normal/braking/overtaking and `side=1/2` means the
detected vehicle is left/right. Appended fields report the left/right scan
confirmation counts and flags, mode/cone/speed readiness, two-lane entry state,
diagnostic lane width, center bias, steering, speed, and an entry
block bitmask. The bitmask is `1=disabled`, `2=mode`, `4=cone`, `8=straight`,
`16=speed`, and `32=no confirmed vehicle`. The main tuning launch arguments are
`overtake_min_points`, `overtake_confirm_scans`,
`overtake_wall_analysis_forward_max`,
`overtake_wall_analysis_lateral_limit`, `overtake_vehicle_max_span`,
`overtake_wall_reject_span`, `overtake_brake_speed`, `overtake_drive_speed`,
and `overtake_min_entry_speed`. The retained center-bias, steering-angle, and
lane-width thresholds are diagnostic references only.

## Keyboard pause/resume

Run the main driving launch in one terminal. In a second interactive terminal,
start the keyboard safety node:

```bash
source ~/xycar_ws/install/setup.bash
ros2 run final_drive keyboard_stop
```

The keyboard node always starts in `STOP`. Press `SPACE` to run, and press
`SPACE` again to stop. `S` forces stop, `R` requests run, and `Q` sends stop
before exiting. Keep this terminal focused while driving. If its heartbeat is
lost for 0.5 seconds, `drive_supervisor` stops the vehicle and requires a new
stop-then-run sequence before motion can resume. The command and authoritative
state topics are `/final_drive/manual_stop` and
`/final_drive/manual_stop_state`.

Keyboard control is required by the main launch by default, so the vehicle
cannot move until the keyboard node has sent its initial `STOP` handshake and
the operator presses `SPACE`. For an intentional non-driving bench test without
the keyboard node, pass `require_keyboard_control:=false`.

The launch also starts the traffic vision node built inside `final_drive`.
Traffic-light detection uses `Y=0..239` of the default 640x480 image. Pixels at
`Y>=240` are masked and any box crossing that boundary is rejected. With the
70-percent horizontal ROI, the active signal area is `x=96..543`, `y=0..239`.
The checkerboard implementation is preserved in
`final_drive_vision/checkerboard_detector.py`, but `enable_checkerboard` is
`false` in the integrated launch. Therefore the live node does not create the
detector, run its image calculations, or publish `/checkerboard_detected`.
Offline checkerboard tools remain available for later use.
`drive_supervisor` uses a safety-asymmetric confirmation rule. Two consecutive
structured red (`1`) or yellow (`2`) detections latch a full stop in both
CAMERA and CONE modes. Green (`3`) and green+left (`4`) still require three
consecutive frames before they release that latch and keep the selected driver
going straight; left-turn steering is intentionally not implemented yet. State
`0` resets an unfinished green confirmation but cannot release a confirmed
stop. This rejects a one-frame false red while preventing a brief false green
from restarting the vehicle. The confirmed state is available on
`/final_drive/traffic_state`.

For safety, the supervisor also stops if `/traffic_light_state` has not arrived
for 0.75 seconds. At startup, the vehicle remains stopped through `NONE`, red,
yellow, and green+left states. Only three consecutive plain-green (`3`) frames
release the one-time startup latch. This prevents the detector's early `NONE`
frames or the first unconfirmed red frame from allowing a brief launch. After
that first release, the normal two-frame red/yellow stop and three-frame green
restart rules apply. Set `require_green_to_start:=false` only for an intentional
bench test that does not begin at the traffic light. Set
`require_traffic_for_motion:=false` only when intentionally testing without the
traffic detector.

The vehicle starts in `CAMERA` mode. `drive_supervisor` only recognizes a cone
entrance when shape-validated objects are visible on both sides at the same
forward position. Their width must be 0.70--0.98 m, each object must lie near an
expected boundary, and the pair center must remain close to the vehicle center.
Same-side rows never cause pre-deceleration or a mode change. A valid bilateral
gate caps camera-mode speed at 8.0; `CONE` is selected only after the cone node
also reports reliable tracked lanes on both sides. The last valid bilateral
gate is remembered for 0.80 seconds (`cone_bilateral_hold_sec`) so the cone
tracker has time to complete its multi-scan confirmation. One-sided detections
do not refresh this memory.

The cone node publishes `TRACK`, `RECOVERY`, or `STOP` as compact numeric state
on `/final_drive/cone_lane_status` (`1=one-side track`, `2=both-side track`,
`3=recovery`, `0=stop`). Brief
occlusions remain under cone recovery; once that recovery declares `STOP`, the
supervisor returns to `CAMERA` without waiting for the cone speed ramp to reach
zero. Multiple cone sections are enabled by default so the same entrance can be
used again on every lap. The one-second post-exit cooldown only suppresses
immediate end-of-section flicker; it does not lock later re-entry.

The supervisor also keeps the motor stopped until the first fresh `/scan`
message arrives. This prevents the camera node from starting the vehicle while
the LiDAR driver is still warming up. `/final_drive/cone_status` is published
by the supervisor for logging (`0=none`, `1=approach`, `2=cone command active`).

## Build and run on the vehicle

```bash
cd ~/xycar_ws
colcon build --symlink-install --packages-select final_drive
source install/setup.bash
ros2 launch final_drive final_drive.launch.py
```

To run the track without the separate space-bar controller while still waiting
for the initial confirmed green light:

```bash
ros2 launch final_drive final_drive.launch.py require_keyboard_control:=false
```

Traffic perception can still be run by itself from the same package:

```bash
ros2 launch final_drive camera_traffic_vision.launch.py
ros2 launch final_drive cone_drive.launch.py
ros2 run final_drive traffic_bag_viewer /path/to/Xytron
ros2 run final_drive traffic_bag_evaluate /path/to/traffic_bags --stride 10
```

## Preparing the recorded traffic-light dataset

`traffic_dataset_prepare` converts the fifteen `traffic_red_*`,
`traffic_yellow_*`, `traffic_green_*`, `traffic_leftgreen_*`, and
`traffic_none_*` image bags into a five-class YOLO dataset. The supplied
sequences use two checked signal-housing boxes per bag. Sparse optical flow
propagates them over the approach, and smoothly blends the middle and near
tracks so the foreground rail cannot pull the close-range box away from the
signal. Only the per-sequence frame intervals reviewed in the original-quality
viewer are exported. Earlier, later, or ambiguous frames are excluded rather
than mislabeled as empty-background examples.

Run this offline; it is not part of the driving launch:

```bash
source ~/xycar_ws/install/setup.bash
ros2 run final_drive traffic_dataset_prepare \
  /path/to/traffic_rosbag /path/to/traffic_dataset_v1 --stride 6
```

The output contains `images/{train,val}`, `labels/{train,val}`, `data.yaml`,
`anchors.json`, and `manifest.json`. Bags ending in `_01` or `_02` are training
data, while `_03` bags are validation data. This bag-level split prevents
nearly identical adjacent frames from appearing on both sides of validation.
The five classes are `red`, `yellow`, `green`, `left_green`, and `off`.

This command only prepares and labels the archived v1 experiment. The live
node uses the separately trained v2 housing model described below.

### Recommended v2 housing detector

The v1 experiment showed that YOLO can localize the four-lamp housing at the
requested distance, but direct five-state classification confuses red with
yellow and green with left-green. The recommended v2 profile therefore trains
one `traffic_light` housing class. Lamp state is classified later inside the
detected crop, where the four fixed lamp positions make HSV/brightness tests
more reliable.

Build the v2 dataset with confirmed empty frames from the signal sequences and
the reviewed `run_04_cone` driving bag:

```bash
ros2 run final_drive traffic_housing_dataset_prepare \
  /path/to/traffic_rosbag /path/to/traffic_housing_dataset_v2 \
  --driving-background-bag /path/to/rosbag/run_04_cone
```

The supplied configuration exports 291 positive signal frames and 74
empty-label backgrounds. The backgrounds include cones, a stopped vehicle,
walls, ceiling lights, rails, corners, and frames after the signal has fully
left the image. The original five-state dataset and model remain separate.

The live traffic node now uses the installed
`models/traffic_housing_v2_best.onnx` detector followed by the second-stage
classifier in
`final_drive_vision/housing_classifier.py`. It divides a YOLO housing box into
the fixed `R/Y/L/G` order and measures the brightest 20 percent of each lamp
disk. Illuminated LEDs are clipped nearly white by this camera, so slot
brightness is the primary signal; hue is not used as the primary decision.
The current threshold (`V >= 240`) classified all 221 training crops and all 70
YOLO validation crops correctly. The supervisor's two-frame red/yellow rule is
still required because reflections can create an isolated bright slot.

To inspect individual YOLO predictions and their four-slot decisions:

```bash
ros2 run final_drive traffic_housing_color_visualize \
  --images /path/to/traffic_housing_dataset_v2/images/val \
  --model $(ros2 pkg prefix final_drive)/share/final_drive/models/traffic_housing_v2_best.onnx \
  --output /tmp/traffic_slots \
  traffic_red_03_000102 traffic_yellow_03_000090
```

The model is a fixed-size legacy-export ONNX (`imgsz=640`, `opset=12`, no
embedded NMS). The legacy TorchScript export is intentional: ONNX graphs from
the PyTorch 2.11 dynamo exporter do not load in the vehicle's OpenCV DNN. At
startup the traffic node loads the installed model and fails closed if OpenCV
cannot parse it. The image subscription keeps only the newest frame so slow
CPU inference cannot accumulate an old-frame queue and delay a red decision.

The default stop/go confirmations are two and three consecutive frames, and all
debug windows are disabled. `traffic_confirm_frames` only changes the green
release threshold. For a faster two-frame release, or to explicitly enable the
traffic debug window, use:

```bash
ros2 launch final_drive final_drive.launch.py \
  traffic_confirm_frames:=2 traffic_debug:=true
```

Watch the selected mode:

```bash
ros2 topic echo /final_drive/mode
ros2 topic echo /traffic_light_state
ros2 topic echo /final_drive/traffic_state
```

For rosbag playback, do not start the physical sensors. If the bag contains a
recorded `/xycar_motor` topic, remap it so it cannot compete with the supervisor:

```bash
ros2 launch final_drive final_drive.launch.py start_camera:=false start_lidar:=false
ros2 bag play <bag-folder> --remap /xycar_motor:=/recorded_xycar_motor
```

If the bag already contains `/traffic_light_state`, also pass
`start_traffic_vision:=false` so the recorded and live detector topics do not
compete. If the bag only contains `/image_raw`, leave traffic vision enabled.

For the next transition test, record the decision inputs, the compressed camera
image needed to diagnose traffic-light misses, and the final output:

```bash
ros2 bag record -o run_03_control /scan /image_raw/compressed /xycar_motor \
  /final_drive/camera_cmd /final_drive/cone_cmd \
  /final_drive/camera_lane_state /final_drive/overtake_state \
  /final_drive/cone_lane_status \
  /traffic_light_state /final_drive/traffic_state \
  /final_drive/cone_status /final_drive/mode \
  /final_drive/manual_stop /final_drive/manual_stop_state
```

These internal topics make the exact detection, pre-deceleration, and handoff
times directly measurable in the next bag. Camera lane state is
`[state,left_x,right_x,center_x,width]`, where `0=not found` and `1=accepted`.

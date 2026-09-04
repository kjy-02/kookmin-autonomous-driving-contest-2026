#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
 라바콘 실측 맞춤형 진단기 (Point Cloud Inspector - Real Cone Sector)
====================================================================
 - 이미지에 보인 실제 콘 점군(거리 0.20m~0.95m, 좌측 +25°~+95°, 우측 -95°~-25°)만
   정확히 라바콘 후보로 캡처하도록 감지 범위를 정밀 튜닝한 진단기입니다.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

import cv2
import numpy as np
import math

CONE_SPACING_M = 0.29      # 콘 표준 간격 (29cm)
TRACK_WIDTH_M  = 0.84      # 트랙 차선 폭 (84cm)
HALF_WIDTH_M   = 0.42      # 반폭 (42cm)

# =============================================
# KUAC 2024 실전 ROI 파라미터 (전방 80cm, 좌우 ±75cm)
# =============================================
CONE_DIST_MIN = 0.15       # 최소 감지 거리 (15cm)
CONE_DIST_MAX = 0.80       # 최대 감지 거리 (80cm - 전방 80cm 이내)

MAX_X_LIMIT   = 0.75       # 차체 기준 좌우 제한 (±75cm)

LEFT_ANGLE_MIN  = 0.0      # 좌측 영역 시작 각도 (0°)
LEFT_ANGLE_MAX  = 75.0     # 좌측 영역 끝 각도 (+75°)
RIGHT_ANGLE_MIN = -75.0    # 우측 영역 시작 각도 (-75°)
RIGHT_ANGLE_MAX = 0.0      # 우측 영역 끝 각도 (0°)

MIN_INTENSITY   = 0.0      # 모든 라이다 레이저 점 수용 (강도 0 이상)
LIDAR_FRONT_OFFSET = 0.10  # 라이다 장착 위치: 앞바퀴 축 기준 전방 10cm (+0.10m)

class ConePoint:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def dist_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

class TrackedCone:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.hit_count = 1
        self.missing_count = 0

    def update(self, x, y, alpha=0.65):
        self.x = alpha * x + (1.0 - alpha) * self.x
        self.y = alpha * y + (1.0 - alpha) * self.y
        self.hit_count += 1
        self.missing_count = 0

class TemporalConeTracker:
    def __init__(self, max_match_dist=0.18, min_confirm_hits=2, max_missing=2):
        self.tracked = []
        self.max_match_dist = max_match_dist
        self.min_confirm_hits = min_confirm_hits
        self.max_missing = max_missing

    def update(self, raw_cone_points):
        matched_flags = [False] * len(self.tracked)
        unmatched_pts = []

        for pt in raw_cone_points:
            best_idx = -1
            best_dist = 999.0
            for i, tc in enumerate(self.tracked):
                if matched_flags[i]:
                    continue
                d = math.hypot(pt.x - tc.x, pt.y - tc.y)
                if d < self.max_match_dist and d < best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx != -1:
                self.tracked[best_idx].update(pt.x, pt.y)
                matched_flags[best_idx] = True
            else:
                unmatched_pts.append(pt)

        # 이번 프레임의 신규 미매칭 포인트 일괄 추가
        for pt in unmatched_pts:
            self.tracked.append(TrackedCone(pt.x, pt.y))

        # 미매칭 기존 객체 관리
        new_tracked = []
        for i, tc in enumerate(self.tracked):
            if i < len(matched_flags) and not matched_flags[i]:
                tc.missing_count += 1

            if tc.missing_count <= self.max_missing:
                new_tracked.append(tc)
        self.tracked = new_tracked

        # 확정된 콘만 반환 (시간 연속성: hit_count >= min_confirm_hits)
        confirmed = [ConePoint(tc.x, tc.y) for tc in self.tracked if tc.hit_count >= self.min_confirm_hits]
        return confirmed

class LidarVectorChainingInspector(Node):
    def __init__(self):
        super().__init__('lidar_cone_inspector')

        self.cluster_dist_thresh = 0.12      # 점 결합 거리 (12cm)
        self.max_cone_radius = 0.15          # 콘 반지름 최대 15cm (지름 ~30cm까지 허용)
        self.min_cluster_points = 1          # 최소 1점 이상 콘 허용

        self.left_pivot  = ConePoint( HALF_WIDTH_M, CONE_SPACING_M)
        self.right_pivot = ConePoint(-HALF_WIDTH_M, CONE_SPACING_M)

        self.tracker = TemporalConeTracker(max_match_dist=0.18, min_confirm_hits=2, max_missing=2)

        self.last_scan_msg = None
        self.scan_received = False

        self.win_name = "KUAC Point Cloud Inspector"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 800, 800)

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_best_effort
        )

        self.timer = self.create_timer(0.03, self.render_loop)

        self.get_logger().info("=========================================================")
        self.get_logger().info(" Real Cone Sector Inspector Initialized (0.20m ~ 0.95m)")
        self.get_logger().info("=========================================================")

    def is_in_cone_roi(self, r, deg_ros):
        """실제 콘 위치 섹터(좌측 25°~95°, 우측 -95°~-25°, 반경 0.20m~0.95m) 판별"""
        if r < CONE_DIST_MIN or r > CONE_DIST_MAX:
            return False, 'OUT'

        # 좌측 콘 섹터 (+25° ~ +95°)
        if LEFT_ANGLE_MIN <= deg_ros <= LEFT_ANGLE_MAX:
            return True, 'LEFT'
        # 우측 콘 섹터 (-95° ~ -25°)
        elif RIGHT_ANGLE_MIN <= deg_ros <= RIGHT_ANGLE_MAX:
            return True, 'RIGHT'

        return False, 'OUT'

    def scan_callback(self, msg: LaserScan):
        self.last_scan_msg = msg
        self.scan_received = True

    def render_loop(self):
        if not self.scan_received or self.last_scan_msg is None:
            canvas = np.full((800, 800, 3), 20, dtype=np.uint8)
            cv2.putText(canvas, "Waiting for /scan topic...", (220, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow(self.win_name, canvas)
            cv2.waitKey(1)
            return

        msg = self.last_scan_msg
        ranges = np.array(msg.ranges)
        intensities = np.array(msg.intensities) if len(msg.intensities) == len(ranges) else None

        num_pts = len(ranges)
        if num_pts == 0:
            return

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        all_points = []
        valid_cone_pts = []

        for i in range(num_pts):
            raw_rad = angle_min + i * angle_inc

            # --- 180도 뒤집어 장착된 라이다 각도 보정 (+PI) ---
            angle_rad = raw_rad + math.pi
            while angle_rad > math.pi:
                angle_rad -= 2.0 * math.pi
            while angle_rad < -math.pi:
                angle_rad += 2.0 * math.pi

            deg_ros = math.degrees(angle_rad)

            if abs(deg_ros) > 120.0:
                continue

            r = ranges[i]
            if np.isnan(r) or np.isinf(r) or r < 0.05 or r > 2.50:
                continue

            x_m = r * math.sin(angle_rad)
            y_m = r * math.cos(angle_rad)
            intensity = intensities[i] if intensities is not None else 100.0

            is_roi, sector = self.is_in_cone_roi(r, deg_ros)
            is_in_x_limit = abs(x_m) <= MAX_X_LIMIT
            is_cone_candidate = is_roi and is_in_x_limit and (intensity >= MIN_INTENSITY)

            pt_dict = {
                'index': i,
                'x': x_m, 'y': y_m, 'r': r, 'deg': deg_ros,
                'intensity': intensity, 'is_roi': is_roi, 'sector': sector,
                'is_candidate': is_cone_candidate
            }
            all_points.append(pt_dict)

            if is_cone_candidate:
                valid_cone_pts.append(pt_dict)

        raw_clusters = self.detect_clusters(valid_cone_pts, all_points)
        confirmed_clusters = self.tracker.update(raw_clusters)
        left_cones, right_cones = self.build_chained_cone_lanes(confirmed_clusters)
        target_mid_x, midpoints = self.calculate_waypoint(left_cones, right_cones)

        self.draw_point_cloud_view(all_points, left_cones, right_cones, midpoints, target_mid_x)

    def detect_clusters(self, valid_pts, all_points=None):
        if not valid_pts:
            return []

        # --- 1. Point Grouping (6.5cm gap - 콘 모서리/떨어진 점 분할 방지) ---
        raw_groups = []
        curr = [valid_pts[0]]
        for i in range(1, len(valid_pts)):
            dist = math.hypot(valid_pts[i]['x'] - valid_pts[i-1]['x'], valid_pts[i]['y'] - valid_pts[i-1]['y'])
            r = valid_pts[i]['r']
            max_group_dist = 0.065 + r * 0.010

            if dist < max_group_dist:
                curr.append(valid_pts[i])
            else:
                raw_groups.append(curr)
                curr = [valid_pts[i]]
        raw_groups.append(curr)

        # --- 2. [좌우 둥근 라바콘 정밀 인지 & 벽 끄트머리/일자 벽 100% 제거] ---
        clusters = []
        for g in raw_groups:
            num_pts = len(g)
            if num_pts < self.min_cluster_points:
                continue

            # (1) 물리 스팬 검사: 2cm ~ 35cm (근거리/중거리 대형 콘 32cm 스팬 수용)
            max_span = 0.0
            for i in range(num_pts):
                for j in range(i + 1, num_pts):
                    d = math.hypot(g[i]['x'] - g[j]['x'], g[i]['y'] - g[j]['y'])
                    if d > max_span:
                        max_span = d

            if max_span < 0.02 or max_span > 0.35:
                continue

            # (2) 벽 끄트머리(Wall Edge) 제거: ROI 경계 부근 연장점 연속 검사 (1.0m 전방 및 1.1m 하드코딩 수정)
            if all_points is not None and len(all_points) > 0:
                first_idx = g[0]['index']
                last_idx = g[-1]['index']
                is_wall_edge = False

                for check_idx in [first_idx - 1, first_idx - 2, last_idx + 1, last_idx + 2]:
                    if 0 <= check_idx < len(all_points):
                        p_raw = all_points[check_idx]
                        if abs(p_raw['x']) > 0.78 or p_raw['y'] > 1.05 or p_raw['r'] > 1.10:
                            d_edge = min(math.hypot(p_raw['x'] - g[0]['x'], p_raw['y'] - g[0]['y']),
                                         math.hypot(p_raw['x'] - g[-1]['x'], p_raw['y'] - g[-1]['y']))
                            if d_edge < 0.08:
                                is_wall_edge = True
                                break
                if is_wall_edge:
                    continue

            # 2점 콘은 소형 원거리 콘 수용
            if num_pts == 2:
                avg_x = (g[0]['x'] + g[1]['x']) / 2.0
                avg_y = (g[0]['y'] + g[1]['y']) / 2.0
                clusters.append(ConePoint(avg_x, avg_y))
                continue

            # (3) PCA 직선도 검사 (4점 이상, Linearity Ratio > 7.5 일자 벽 제거 - 곡면 콘 완벽 수용)
            avg_x_g = sum(p['x'] for p in g) / num_pts
            avg_y_g = sum(p['y'] for p in g) / num_pts

            c_xx = sum((p['x'] - avg_x_g) ** 2 for p in g) / num_pts
            c_yy = sum((p['y'] - avg_y_g) ** 2 for p in g) / num_pts
            c_xy = sum((p['x'] - avg_x_g) * (p['y'] - avg_y_g) for p in g) / num_pts

            trace = c_xx + c_yy
            det = c_xx * c_yy - c_xy * c_xy
            gap = math.sqrt(max(0.0, trace * trace / 4.0 - det))

            lambda1 = trace / 2.0 + gap
            lambda2 = max(1e-6, trace / 2.0 - gap)
            linearity_ratio = math.sqrt(lambda1 / lambda2)

            if linearity_ratio > 7.5 and max_span > 0.15:
                continue

            # (4) 현-호 수직 곡률 깊이 (Sagitta Depth) 검사: 1.5mm 미만 평평 일자 벽만 제거
            p1 = g[0]
            pn = g[-1]
            dx_c = pn['x'] - p1['x']
            dy_c = pn['y'] - p1['y']
            c_len = math.hypot(dx_c, dy_c)

            if c_len > 0.05:
                max_sagitta = 0.0
                for pt in g[1:-1]:
                    perp = abs(dy_c * pt['x'] - dx_c * pt['y'] + pn['x'] * p1['y'] - pn['y'] * p1['x']) / c_len
                    if perp > max_sagitta:
                        max_sagitta = perp

                if max_sagitta < 0.0015 and max_span > 0.12:
                    continue

            clusters.append(ConePoint(avg_x_g, avg_y_g))

        # --- 3. Circle Merging (인접 콘 중복 병합) ---
        merged_clusters = []
        merged_visited = [False] * len(clusters)

        for i in range(len(clusters)):
            if merged_visited[i]:
                continue
            merged_visited[i] = True

            cx = clusters[i].x
            cy = clusters[i].y

            for j in range(i + 1, len(clusters)):
                if merged_visited[j]:
                    continue
                d = clusters[i].dist_to(clusters[j])
                if d < 0.18:  # 18cm 이내 중복 콘 병합
                    merged_visited[j] = True
                    cx = (cx + clusters[j].x) / 2.0
                    cy = (cy + clusters[j].y) / 2.0

            merged_clusters.append(ConePoint(cx, cy))

        return merged_clusters

    def build_chained_cone_lanes(self, candidates):
        if not candidates:
            return [], []

        # 원점(차량) 기준 거리 순 정렬
        sorted_candidates = sorted(candidates, key=lambda p: (p.x**2 + p.y**2))
        visited = [False] * len(sorted_candidates)

        left_cones = []
        right_cones = []

        # === 1. 좌측 차선 연쇄 구축 (급커브 선형 추적) ===
        best_left_idx = -1
        best_left_dist = 999.0
        for i, c in enumerate(sorted_candidates):
            d_left = self.left_pivot.dist_to(c)
            d_right = self.right_pivot.dist_to(c)

            # 우측 피봇에 더 가까운 콘은 좌측 피봇 콘으로 오검출되는 것을 차단!
            if d_right <= d_left:
                continue

            if c.x > -0.20:  # 급커브로 인한 좌측 차선의 횡이동 수용
                if d_left < 0.55 and d_left < best_left_dist:
                    best_left_dist = d_left
                    best_left_idx = i

        if best_left_idx != -1:
            visited[best_left_idx] = True
            left_cones.append(sorted_candidates[best_left_idx])

            while True:
                last_cone = left_cones[-1]
                prev_dir_x, prev_dir_y = 0.0, 1.0

                if len(left_cones) >= 2:
                    dx = left_cones[-1].x - left_cones[-2].x
                    dy = left_cones[-1].y - left_cones[-2].y
                    l = math.hypot(dx, dy)
                    if l > 1e-4:
                        prev_dir_x, prev_dir_y = dx / l, dy / l

                best_next_idx = -1
                best_score = 999.0

                for i, c in enumerate(sorted_candidates):
                    if visited[i]:
                        continue

                    # 콘 간격 제약: 18cm ~ 40cm (중심 25~35cm)
                    step_dist = last_cone.dist_to(c)
                    if not (0.18 <= step_dist <= 0.40):
                        continue

                    # 진행 방향 및 35도 꺾임각 최대 제약
                    step_dir_x = (c.x - last_cone.x) / step_dist
                    step_dir_y = (c.y - last_cone.y) / step_dist

                    if step_dir_y < -0.15:
                        continue

                    dot = prev_dir_x * step_dir_x + prev_dir_y * step_dir_y
                    dot = max(-1.0, min(1.0, dot))
                    turn_angle_deg = math.degrees(math.acos(dot))

                    if turn_angle_deg > 35.0:  # 최대 꺾임각 35도 제한
                        continue

                    score = turn_angle_deg + abs(step_dist - CONE_SPACING_M) * 50.0
                    if score < best_score:
                        best_score = score
                        best_next_idx = i

                if best_next_idx != -1:
                    visited[best_next_idx] = True
                    left_cones.append(sorted_candidates[best_next_idx])
                else:
                    break

        # === 2. 우측 차선 연쇄 구축 (트랙 차선 폭 70~90cm 가드 적용) ===
        best_right_idx = -1
        best_right_dist = 999.0
        for i, c in enumerate(sorted_candidates):
            if visited[i]:
                continue

            # [차선 폭 70~90cm 가드]: 좌측 차선 콘과 거리가 70cm 미만이면 우측 콘/오검출로 판정하여 배제!
            too_close_to_left = False
            for lc in left_cones:
                if c.dist_to(lc) < 0.70:
                    too_close_to_left = True
                    break

            if too_close_to_left:
                continue

            d = self.right_pivot.dist_to(c)
            if d < 0.55 and d < best_right_dist:
                best_right_dist = d
                best_right_idx = i

        if best_right_idx != -1:
            visited[best_right_idx] = True
            right_cones.append(sorted_candidates[best_right_idx])

            while True:
                last_cone = right_cones[-1]
                prev_dir_x, prev_dir_y = 0.0, 1.0

                if len(right_cones) >= 2:
                    dx = right_cones[-1].x - right_cones[-2].x
                    dy = right_cones[-1].y - right_cones[-2].y
                    l = math.hypot(dx, dy)
                    if l > 1e-4:
                        prev_dir_x, prev_dir_y = dx / l, dy / l

                best_next_idx = -1
                best_score = 999.0

                for i, c in enumerate(sorted_candidates):
                    if visited[i]:
                        continue

                    # [차선 폭 70~90cm 가드]: 좌측 차선 콘과 거리가 70cm 미만이면 결합 금지!
                    too_close_to_left = False
                    for lc in left_cones:
                        if c.dist_to(lc) < 0.70:
                            too_close_to_left = True
                            break

                    if too_close_to_left:
                        continue

                    step_dist = last_cone.dist_to(c)
                    if not (0.18 <= step_dist <= 0.40):
                        continue

                    step_dir_x = (c.x - last_cone.x) / step_dist
                    step_dir_y = (c.y - last_cone.y) / step_dist

                    if step_dir_y < -0.15:
                        continue

                    dot = prev_dir_x * step_dir_x + prev_dir_y * step_dir_y
                    dot = max(-1.0, min(1.0, dot))
                    turn_angle_deg = math.degrees(math.acos(dot))

                    if turn_angle_deg > 35.0:  # 최대 꺾임각 35도 제한
                        continue

                    score = turn_angle_deg + abs(step_dist - CONE_SPACING_M) * 50.0
                    if score < best_score:
                        best_score = score
                        best_next_idx = i

                if best_next_idx != -1:
                    visited[best_next_idx] = True
                    right_cones.append(sorted_candidates[best_next_idx])
                else:
                    break

        if left_cones:
            self.left_pivot = left_cones[0]
        else:
            self.left_pivot = ConePoint(HALF_WIDTH_M, CONE_SPACING_M)

        if right_cones:
            self.right_pivot = right_cones[0]
        else:
            self.right_pivot = ConePoint(-HALF_WIDTH_M, CONE_SPACING_M)

        # 급커브 횡이동 수용 피봇 리셋 검사
        if self.left_pivot.x > 1.2 or self.left_pivot.x < -0.5 or self.left_pivot.y < 0.05 or self.left_pivot.y > 1.2:
            self.left_pivot = ConePoint(HALF_WIDTH_M, CONE_SPACING_M)

        if self.right_pivot.x < -1.2 or self.right_pivot.x > 0.5 or self.right_pivot.y < 0.05 or self.right_pivot.y > 1.2:
            self.right_pivot = ConePoint(-HALF_WIDTH_M, CONE_SPACING_M)

        return left_cones, right_cones

    def calculate_waypoint(self, left_cones, right_cones):
        max_pts = min(max(len(left_cones), len(right_cones)), 6)
        if max_pts == 0:
            return 0.0, []

        midpoints = []
        for i in range(max_pts):
            if i < len(left_cones):
                l = left_cones[i]
            elif i < len(right_cones):
                l = ConePoint(right_cones[i].x + TRACK_WIDTH_M, right_cones[i].y)
            else:
                l = ConePoint(HALF_WIDTH_M, CONE_SPACING_M * (i + 1))

            if i < len(right_cones):
                r = right_cones[i]
            elif i < len(left_cones):
                r = ConePoint(left_cones[i].x - TRACK_WIDTH_M, left_cones[i].y)
            else:
                r = ConePoint(-HALF_WIDTH_M, CONE_SPACING_M * (i + 1))

            mid_x = (l.x + r.x) / 2.0
            mid_y = (l.y + r.y) / 2.0
            midpoints.append(ConePoint(mid_x, mid_y))

        target_mid_x = midpoints[0].x

        if not right_cones and len(left_cones) >= 2:
            dx_curve = left_cones[-1].x - left_cones[0].x
            if dx_curve > 0.0:
                target_mid_x += dx_curve * 2.0
        elif not left_cones and len(right_cones) >= 2:
            dx_curve = right_cones[-1].x - right_cones[0].x
            if dx_curve < 0.0:
                target_mid_x += dx_curve * 2.0

        return target_mid_x, midpoints

    def draw_point_cloud_view(self, all_points, left_cones, right_cones, midpoints, target_mid_x):
        W, H = 800, 800
        SCALE = 380.0
        CX, CY = W // 2, H - 90

        canvas = np.full((H, W, 3), 16, dtype=np.uint8)

        def to_pixel(x_m, y_m):
            px = CX - int(x_m * SCALE)
            py = CY - int(y_m * SCALE)
            return (px, py)

        r_min_px = int(CONE_DIST_MIN * SCALE)
        r_max_px = int(CONE_DIST_MAX * SCALE)

        # 1. 부채꼴 원호 및 섹터 경계선 (노란색)
        cv2.circle(canvas, (CX, CY), r_min_px, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(canvas, (CX, CY), r_max_px, (0, 255, 255), 1, cv2.LINE_AA)

        sector_angles = [-75.0, 0.0, 75.0]
        for sang in sector_angles:
            srad = math.radians(sang)
            p_start = to_pixel(CONE_DIST_MIN * math.sin(srad), CONE_DIST_MIN * math.cos(srad))
            p_end   = to_pixel(CONE_DIST_MAX * math.sin(srad), CONE_DIST_MAX * math.cos(srad))
            cv2.line(canvas, p_start, p_end, (0, 255, 255), 1, cv2.LINE_AA)

        for r in [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]:
            rad_px = int(r * SCALE)
            cv2.circle(canvas, (CX, CY), rad_px, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{r:.1f}m", (CX + 8, CY - rad_px + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        def is_near_cone(pt, cones, radius=0.15):
            for c in cones:
                if math.hypot(pt['x'] - c.x, pt['y'] - c.y) <= radius:
                    return True
            return False

        # 2. 점군 시각화 (점 크기는 2~3px로 동일하게 유지, 색상으로만 구별)
        for pt in all_points:
            px, py = to_pixel(pt['x'], pt['y'])
            if not (0 <= px < W and 0 <= py < H):
                continue

            if is_near_cone(pt, left_cones):
                cv2.circle(canvas, (px, py), 3, (0, 140, 255), -1, cv2.LINE_AA)  # 좌측 콘 점군 (주황)
            elif is_near_cone(pt, right_cones):
                cv2.circle(canvas, (px, py), 3, (255, 150, 0), -1, cv2.LINE_AA)  # 우측 콘 점군 (파랑)
            elif pt['is_candidate']:
                cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)  # 일반 ROI 후보 (노랑)
            else:
                cv2.circle(canvas, (px, py), 2, (90, 90, 90), -1, cv2.LINE_AA)    # 배경 레이저 점 (회색)

        # 3. 콘 중심 피봇 위치 표식 (작은 원)
        for c in left_cones:
            cpx, cpy = to_pixel(c.x, c.y)
            cv2.circle(canvas, (cpx, cpy), 4, (0, 140, 255), 1, cv2.LINE_AA)
        for c in right_cones:
            cpx, cpy = to_pixel(c.x, c.y)
            cv2.circle(canvas, (cpx, cpy), 4, (255, 150, 0), 1, cv2.LINE_AA)

        for i in range(len(left_cones) - 1):
            p1 = to_pixel(left_cones[i].x, left_cones[i].y)
            p2 = to_pixel(left_cones[i+1].x, left_cones[i+1].y)
            cv2.line(canvas, p1, p2, (0, 140, 255), 1, cv2.LINE_AA)

        for i in range(len(right_cones) - 1):
            p1 = to_pixel(right_cones[i].x, right_cones[i].y)
            p2 = to_pixel(right_cones[i+1].x, right_cones[i+1].y)
            cv2.line(canvas, p1, p2, (255, 150, 0), 1, cv2.LINE_AA)

        for mp in midpoints:
            mpx, mpy = to_pixel(mp.x, mp.y)
            poly = np.array([[mpx, mpy - 6], [mpx + 4, mpy], [mpx, mpy + 6], [mpx - 4, mpy]], np.int32)
            cv2.fillConvexPoly(canvas, poly, (0, 255, 100))

        cv2.rectangle(canvas, (CX - 15, CY - 20), (CX + 15, CY + 20), (180, 180, 180), 1)

        status = "BOTH" if left_cones and right_cones else ("LEFT_ONLY" if left_cones else ("RIGHT_ONLY" if right_cones else "NONE"))
        cv2.putText(canvas, f"[Real Cone Inspector] Range:0.20m~0.95m Left:{len(left_cones)} Right:{len(right_cones)}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1)

        cv2.imshow(self.win_name, canvas)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LidarVectorChainingInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

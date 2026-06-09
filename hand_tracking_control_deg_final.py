import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import rbpodo as rb
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────
# 네트워크 설정
# ──────────────────────────────────────────
ROBOT_IP = "192.168.0.23"

# ──────────────────────────────────────────
# 좌표계 변환 (카메라 → 로봇 베이스)
# ──────────────────────────────────────────
R_cam_to_robot = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0, -1,  0]
], dtype=float)

t = np.array([0, -1300, 200], dtype=float)   # 카메라 원점 기준 로봇 베이스까지의 오프셋 (mm)

T = np.eye(4)
T[:3, :3] = R_cam_to_robot
T[:3, 3] = t

# ──────────────────────────────────────────
# 로봇 자세 설정
# ──────────────────────────────────────────
R_home = Rotation.from_euler('xyz', [90, 0, 90], degrees=True).as_matrix()  # 기본 엔드이펙터 자세

WORKSPACE_LIMIT = {          # 로봇 작업 범위 (안전 클리핑용, mm)
    "x": (-500, 500),
    "y": (-1000, 1000),
    "z": (100, 700),
}

# ──────────────────────────────────────────
# 서보 제어 파라미터 (move_servo_l)
# ──────────────────────────────────────────
SERVO_T1    = 0.5   # 목표 위치까지 도달 시간 (s)
SERVO_T2    = 0.1   # 목표 위치 유지 시간 (s)
SERVO_GAIN  = 1.0   # 속도 추종률
SERVO_ALPHA = 1.0   # 저역통과 필터 계수

# ──────────────────────────────────────────
# 엔드이펙터 고정 회전값 (서보 시작 전 기본값)
# ──────────────────────────────────────────
FIXED_RX = 90.0
FIXED_RY =  0.0
FIXED_RZ = 90.0

# ──────────────────────────────────────────
# 첫 위치 안정화 설정
# ──────────────────────────────────────────
STABLE_THRESHOLD = 20.0   # 안정으로 판단하는 최대 이동 거리 (mm)
STABLE_DURATION  = 3.0    # 안정 상태를 유지해야 하는 시간 (s)

# ──────────────────────────────────────────
# 노이즈 필터 설정
# ──────────────────────────────────────────
MAX_JUMP      = 100.0   # 프레임 간 허용 최대 이동량, 초과 시 이상값으로 제거 (mm)
EMA_ALPHA     = 0.3     # 위치 EMA 필터 계수 (낮을수록 부드러움, 0~1)
ROT_EMA_ALPHA = 0.3     # 회전 EMA 필터 계수

# ──────────────────────────────────────────
# Depth 카메라 설정
# ──────────────────────────────────────────
MIN_DEPTH    = 0.1   # 유효 depth 최솟값 (m)
MAX_DEPTH    = 2.0   # 유효 depth 최댓값 (m)
DEPTH_RADIUS = 3     # depth 영역 평균 반경, 반경 3 → 7×7 픽셀 영역의 median 사용

# ──────────────────────────────────────────
# 그리퍼 제어 설정
# ──────────────────────────────────────────
GRIP_CLOSE_ANGLE = 90.0    # 손가락 평균 굽힘 각도 이하 → 잡기 (도)
GRIP_OPEN_ANGLE  = 150.0   # 손가락 평균 굽힘 각도 이상 → 놓기 (도)
GRIP_COOLDOWN    = 1.0     # 연속 그리퍼 동작 방지 간격 (s)

# 그리퍼 각도 계산에 사용할 손가락 관절 랜드마크 인덱스 (MCP-PIP-DIP 순)
FINGER_JOINTS = [
    (5,  6,  7),   # 검지
    (9,  10, 11),  # 중지
    (13, 14, 15),  # 약지
    (17, 18, 19),  # 소지
]


# ──────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────

def clip_workspace(pos):
    """로봇 작업 범위를 벗어난 목표 위치를 경계값으로 클리핑"""
    pos[0] = np.clip(pos[0], *WORKSPACE_LIMIT["x"])
    pos[1] = np.clip(pos[1], *WORKSPACE_LIMIT["y"])
    pos[2] = np.clip(pos[2], *WORKSPACE_LIMIT["z"])
    return pos


def get_depth_average(depth_frame, u, v, radius=DEPTH_RADIUS):
    """지정 픽셀 주변 영역의 median depth 반환
    단일 픽셀 대비 outlier에 강건하며 depth 노이즈를 효과적으로 억제
    유효 범위(MIN_DEPTH ~ MAX_DEPTH) 내 픽셀만 사용
    """
    depths = []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            cu = int(np.clip(u + du, 0, depth_frame.width  - 1))
            cv = int(np.clip(v + dv, 0, depth_frame.height - 1))
            d = depth_frame.get_distance(cu, cv)
            if MIN_DEPTH < d < MAX_DEPTH:
                depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def grip(robot, rc, action):
    """공압 그리퍼 제어
    action: "grab" → 잡기, "release" → 놓기
    디지털 출력 비트 조합으로 밸브 개폐
    """
    value = 2 if action == "grab" else 1
    robot.set_dout_bit_combination(rc, 0, 3, value, rb.Endian.LittleEndian)
    time.sleep(0.5)
    robot.set_dout_bit_combination(rc, 0, 3, 0, rb.Endian.LittleEndian)
    time.sleep(0.5)


def get_3d_pos(landmark, depth_frame, intr, w, h):
    """MediaPipe 랜드마크의 3D 위치 반환 (카메라 좌표계, 단위: m)
    depth는 영역 median을 사용하여 노이즈 억제
    유효하지 않은 depth인 경우 None 반환
    """
    u = int(np.clip(landmark.x * w, 0, w - 1))
    v = int(np.clip(landmark.y * h, 0, h - 1))
    z = get_depth_average(depth_frame, u, v)

    if not (MIN_DEPTH < z < MAX_DEPTH):
        return None

    x = (u - intr.ppx) / intr.fx * z
    y = (v - intr.ppy) / intr.fy * z
    return np.array([x, y, -z])   # RealSense z축 반전 (왼손 → 오른손 좌표계)


def normalize(v):
    """벡터 정규화, 영벡터일 경우 그대로 반환"""
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


def calc_joint_angle(a, b, c):
    """세 점 a-b-c의 각도 계산 (b가 꼭짓점, 단위: 도)"""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calc_avg_finger_angle(hand, w, h):
    """4개 손가락 굽힘 각도의 평균 계산 (2D, 단위: 도)
    손 펼침 → 약 150도 이상 / 손 오므림 → 약 90도 이하
    """
    angles = []
    for (a, b, c) in FINGER_JOINTS:
        pa = np.array([hand.landmark[a].x * w, hand.landmark[a].y * h])
        pb = np.array([hand.landmark[b].x * w, hand.landmark[b].y * h])
        pc = np.array([hand.landmark[c].x * w, hand.landmark[c].y * h])
        angles.append(calc_joint_angle(pa, pb, pc))
    return np.mean(angles)


def get_hand_rotation(P0, P5, P9):
    """손바닥 법선벡터 기반 회전행렬 계산 (카메라 좌표계 기준)
    P0: 손목, P5: 검지 MCP, P9: 중지 MCP
    """
    v1 = P5 - P0
    v2 = P9 - P0
    z_axis = normalize(np.cross(v1, v2))
    x_axis = normalize(v1)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def calc_robot_euler(R_current, R_ref):
    """손바닥 회전을 로봇 엔드이펙터 오일러각으로 변환 (xyz, 단위: 도)
    기준 자세(R_ref) 대비 상대 회전을 로봇 좌표계로 변환
    """
    R_rel       = R_current @ R_ref.T
    R_rel_robot = R_cam_to_robot @ R_rel @ R_cam_to_robot.T
    R_final     = R_rel_robot @ R_home
    return Rotation.from_matrix(R_final).as_euler('xyz', degrees=True)


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    # ── 로봇 연결 및 초기화 ──
    print("[로봇] 연결 중...")
    robot = rb.Cobot(ROBOT_IP)
    rc = rb.ResponseCollector()

    robot.set_operation_mode(rc, rb.OperationMode.Real)
    robot.set_speed_bar(rc, 0.5)
    robot.flush(rc)

    grip(robot, rc, "release")
    print("[로봇] 연결 완료 - 그리퍼 열림")

    print("[로봇] 홈 포지션으로 이동 중...")
    robot.move_j(rc, np.array([-90, 0, 90, 0, 90, 90]), 50, 100)
    if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
        robot.wait_for_move_finished(rc)
    rc.error().throw_if_not_empty()
    print("[로봇] 홈 포지션 완료")

    # ── RealSense 카메라 초기화 ──
    print("[카메라] 초기화 중...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)   # depth 프레임을 color 프레임에 정렬
    intr = profile.get_stream(rs.stream.color) \
                  .as_video_stream_profile() \
                  .get_intrinsics()

    time.sleep(1)
    print("[카메라] 초기화 완료 - 바로 시작")

    # ── MediaPipe 손 감지 초기화 ──
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # ── 상태 변수 ──
    first_frame    = True    # True: 첫 위치 안정화 단계 / False: 서보 추종 단계
    stable_start   = None
    stable_pos     = None
    prev_pos       = None
    ema_pos        = None
    R_ref          = None    # 서보 시작 시점에 저장한 손바닥 기준 자세
    ema_euler      = None    # 회전 EMA 필터 상태
    gripper_closed = False
    last_grip_time = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            h, w, _ = color.shape
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(color, hand, mp_hands.HAND_CONNECTIONS)

                # ── 손목 3D 위치 추출 ──
                wrist = hand.landmark[0]
                u = int(np.clip(wrist.x * w, 0, w - 1))
                v = int(np.clip(wrist.y * h, 0, h - 1))
                z = get_depth_average(depth_frame, u, v)   # 영역 median depth

                if MIN_DEPTH < z < MAX_DEPTH:
                    # 카메라 좌표계 → mm 단위 변환
                    Xc = (u - intr.ppx) / intr.fx * z
                    Yc = (v - intr.ppy) / intr.fy * z
                    Zc = -z

                    # 카메라 좌표계 → 로봇 베이스 좌표계 변환
                    Pc = np.array([Xc * 1000, Yc * 1000, Zc * 1000, 1])
                    Pr = T @ Pc

                    # ── 위치 노이즈 필터 ──
                    raw_pos = Pr[:3].copy()

                    # 1단계: 이상값 제거 (프레임 간 이동량이 MAX_JUMP 초과 시 폐기)
                    if prev_pos is not None:
                        jump = np.linalg.norm(raw_pos - prev_pos)
                        if jump > MAX_JUMP:
                            print(f"[필터] 이상값 제거: jump={jump:.1f}mm")
                            raw_pos = prev_pos.copy()
                        else:
                            prev_pos = raw_pos.copy()
                    else:
                        prev_pos = raw_pos.copy()

                    # 2단계: EMA 필터로 고주파 노이즈 억제
                    if ema_pos is None:
                        ema_pos = raw_pos.copy()
                    else:
                        ema_pos = EMA_ALPHA * raw_pos + (1 - EMA_ALPHA) * ema_pos

                    filtered_pos = ema_pos.copy()

                    # ── 회전값 계산 (서보 시작 후에만 수행) ──
                    if not first_frame:
                        P0 = get_3d_pos(hand.landmark[0], depth_frame, intr, w, h)
                        P5 = get_3d_pos(hand.landmark[5], depth_frame, intr, w, h)
                        P9 = get_3d_pos(hand.landmark[9], depth_frame, intr, w, h)

                        if P0 is not None and P5 is not None and P9 is not None and R_ref is not None:
                            R_current = get_hand_rotation(P0, P5, P9)
                            euler = calc_robot_euler(R_current, R_ref)
                            if ema_euler is None:
                                ema_euler = euler.copy()
                            else:
                                ema_euler = ROT_EMA_ALPHA * euler + (1 - ROT_EMA_ALPHA) * ema_euler
                            rx, ry, rz = ema_euler
                        else:
                            # 랜드마크 취득 실패 시 이전 값 유지
                            rx, ry, rz = (ema_euler if ema_euler is not None
                                          else np.array([FIXED_RX, FIXED_RY, FIXED_RZ]))
                    else:
                        rx, ry, rz = FIXED_RX, FIXED_RY, FIXED_RZ

                    target_pos  = clip_workspace(filtered_pos)
                    target_pose = np.array([
                        target_pos[0], target_pos[1], target_pos[2],
                        rx, ry, rz,
                    ])

                    # ── 첫 위치 안정화 (서보 시작 전 단계) ──
                    if first_frame:
                        current_pos = filtered_pos.copy()

                        if stable_pos is None:
                            stable_pos   = current_pos
                            stable_start = time.time()
                            print("[안정화] 손 감지 - 3초 유지 대기 중...")
                        else:
                            dist = np.linalg.norm(current_pos - stable_pos)

                            if dist > STABLE_THRESHOLD:
                                # 손이 움직이면 타이머 리셋
                                stable_pos   = current_pos
                                stable_start = time.time()
                            else:
                                remaining = STABLE_DURATION - (time.time() - stable_start)

                                if remaining > 0:
                                    cv2.putText(color,
                                                f"Hold position: {remaining:.1f}s",
                                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 165, 255), 2)
                                else:
                                    # 안정화 완료 → 첫 위치로 이동
                                    target_pos  = clip_workspace(current_pos)
                                    target_pose = np.array([
                                        target_pos[0], target_pos[1], target_pos[2],
                                        FIXED_RX, FIXED_RY, FIXED_RZ,
                                    ])

                                    print("[로봇] 첫 위치 확정 - 이동 중...")
                                    cv2.putText(color, "Moving to first position...",
                                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 165, 255), 2)
                                    cv2.imshow("Hand Tracking Control", color)
                                    cv2.waitKey(1)

                                    robot.move_l(rc, target_pose, 100, 200)
                                    if robot.wait_for_move_started(rc, 0.1).type() == rb.ReturnType.Success:
                                        robot.wait_for_move_finished(rc)

                                    # 필터 상태를 현재 위치로 초기화
                                    prev_pos = target_pos.copy()
                                    ema_pos  = target_pos.copy()

                                    # 현재 손 자세를 회전 기준으로 저장
                                    P0 = get_3d_pos(hand.landmark[0], depth_frame, intr, w, h)
                                    P5 = get_3d_pos(hand.landmark[5], depth_frame, intr, w, h)
                                    P9 = get_3d_pos(hand.landmark[9], depth_frame, intr, w, h)

                                    if P0 is not None and P5 is not None and P9 is not None:
                                        R_ref     = get_hand_rotation(P0, P5, P9)
                                        ema_euler = np.array([FIXED_RX, FIXED_RY, FIXED_RZ])
                                        print("[회전] 기준 자세 저장 완료")
                                    else:
                                        R_ref = None
                                        ema_euler = np.array([FIXED_RX, FIXED_RY, FIXED_RZ])
                                        print("[회전] 기준 자세 저장 실패 - 고정값으로 진행")

                                    first_frame = False
                                    print("[로봇] 첫 위치 도착 - servo 제어 시작")
                                    robot.disable_waiting_ack(rc)

                    else:
                        # ── 서보 추종 (실시간 위치 명령) ──
                        # 출력 형식: [x, y, z, rx, ry, rz] (mm, 도)
                        # 명령 주기: 카메라 프레임레이트 기준 (~30Hz)
                        robot.move_servo_l(rc, target_pose,
                                           SERVO_T1, SERVO_T2,
                                           SERVO_GAIN, SERVO_ALPHA)

                    # ── 화면 표시 ──
                    if not first_frame:
                        cv2.putText(color, "Tracking",
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                    cv2.putText(color,
                                f"Cam: ({Xc*1000:.0f}, {Yc*1000:.0f}, {Zc*1000:.0f})",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.putText(color,
                                f"Robot: ({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f})",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 200, 255), 2)
                    cv2.putText(color,
                                f"Rot: ({rx:.1f}, {ry:.1f}, {rz:.1f})",
                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 200, 0), 2)
                    cv2.putText(color,
                                f"Raw: ({Pr[0]:.0f}, {Pr[1]:.0f}, {Pr[2]:.0f})",
                                (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)

                # ── 그리퍼 제어 (서보 시작 후에만 수행) ──
                if not first_frame:
                    avg_angle = calc_avg_finger_angle(hand, w, h)
                    now = time.time()
                    cooldown_ok = (now - last_grip_time) > GRIP_COOLDOWN

                    if avg_angle < GRIP_CLOSE_ANGLE and not gripper_closed and cooldown_ok:
                        print(f"[그리퍼] 잡기 (angle={avg_angle:.1f}°)")
                        robot.enable_waiting_ack(rc)
                        grip(robot, rc, "grab")
                        prev_pos  = target_pos.copy()
                        ema_pos   = target_pos.copy()
                        if ema_euler is not None:
                            ema_euler = np.array([rx, ry, rz])
                        gripper_closed = True
                        last_grip_time = now
                        robot.disable_waiting_ack(rc)

                    elif avg_angle > GRIP_OPEN_ANGLE and gripper_closed and cooldown_ok:
                        print(f"[그리퍼] 놓기 (angle={avg_angle:.1f}°)")
                        robot.enable_waiting_ack(rc)
                        grip(robot, rc, "release")
                        prev_pos  = target_pos.copy()
                        ema_pos   = target_pos.copy()
                        if ema_euler is not None:
                            ema_euler = np.array([rx, ry, rz])
                        gripper_closed = False
                        last_grip_time = now
                        robot.disable_waiting_ack(rc)

                    status      = "Closed" if gripper_closed else "Open"
                    color_state = (0, 0, 255) if gripper_closed else (0, 255, 0)
                    cv2.putText(color,
                                f"Gripper: {status} ({avg_angle:.1f}deg)",
                                (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color_state, 2)

            else:
                cv2.putText(color, "Hand not detected",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.imshow("Hand Tracking Control", color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[시스템] 종료")
                break

    except KeyboardInterrupt:
        print("[시스템] 인터럽트로 종료")

    finally:
        robot.enable_waiting_ack(rc)
        grip(robot, rc, "release")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[시스템] 정리 완료")


if __name__ == "__main__":
    main()


import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Optional: only import mediapipe inside functions to keep app startup lean
# import mediapipe as mp

st.set_page_config(
    page_title="Shadow Puppet Parametrisation (MediaPipe + Streamlit)",
    layout="wide",
    page_icon="🖐️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/alobo01/shadow-puppets-streamlit',
        'About': "# Shadow Puppet Viewer\nInteractive hand gesture capture and parametrization using MediaPipe"
    }
)

# --------------------- Custom Styling ---------------------

def apply_custom_css():
    """Apply custom CSS for better visual appeal"""
    st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 3px solid #ff7f0e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ff7f0e;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #e56d00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1em;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Column borders for better separation */
    [data-testid="column"] {
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Json display */
    [data-testid="stJson"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------- Geometry helpers ---------------------

CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),           # thumb
    (0,5), (5,6), (6,7), (7,8),           # index
    (0,9), (9,10), (10,11), (11,12),      # middle
    (0,13), (13,14), (14,15), (15,16),    # ring
    (0,17), (17,18), (18,19), (19,20),    # pinky
    (5,9), (9,13), (13,17)                # palm links
]

FINGER_CHAINS = {
    "T": [1,2,3,4],
    "I": [5,6,7,8],
    "M": [9,10,11,12],
    "R": [13,14,15,16],
    "P": [17,18,19,20],
}

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_u, b_u = unit(a), unit(b)
    val = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

def rotation_to_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix that rotates vector a to vector b."""
    a_u, b_u = unit(a), unit(b)
    v = np.cross(a_u, b_u)
    c = float(np.dot(a_u, b_u))
    if np.linalg.norm(v) < 1e-9:
        # already aligned or opposite
        if c > 0:
            return np.eye(3)
        # 180° flip around any axis orthogonal to a
        axis = unit(np.cross(a_u, np.array([1.0,0.0,0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = unit(np.cross(a_u, np.array([0.0,1.0,0.0])))
        K = np.array([[0, -axis[2], axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return -np.eye(3) + 2*np.outer(axis, axis)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R

def plane_from_points(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
    n = unit(np.cross(q - p, r - p))
    d = float(np.dot(n, p))
    return n, d

def apply_rotation(L: Dict[int, np.ndarray], R: np.ndarray) -> Dict[int, np.ndarray]:
    return {i: R @ L[i] for i in L}

def translate(L: Dict[int, np.ndarray], t: np.ndarray) -> Dict[int, np.ndarray]:
    return {i: L[i] + t for i in L}

def scale(L: Dict[int, np.ndarray], s: float) -> Dict[int, np.ndarray]:
    return {i: L[i] * s for i in L}

# --------------------- Torch & wall ---------------------

@dataclass
class Setup:
    T: np.ndarray            # torch position
    wall_n: np.ndarray       # wall normal (unit)
    wall_d: float            # plane offset (n^T x = d)
    base_radius: float

DEFAULT_SETUP = Setup(
    T=np.array([0.0, 0.0, -2.0], dtype=float),
    wall_n=unit(np.array([0.0, 0.0, 1.0], dtype=float)),
    wall_d=4.0,
    base_radius=2.0
)

def project_point_to_plane_from_T(X: np.ndarray, setup: Setup) -> np.ndarray:
    num = (setup.wall_d - np.dot(setup.wall_n, setup.T))
    den = np.dot(setup.wall_n, X - setup.T)
    if abs(den) < 1e-9:
        return X.copy()
    alpha = num / den
    return setup.T + alpha * (X - setup.T)

# --------------------- Parametrisation ---------------------

def compute_palm_plane(L: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
    return plane_from_points(L[0], L[5], L[17])

def compute_thumb_plane(L: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
    C = 0.5*(L[9] + L[13])
    return plane_from_points(L[1], L[2], C)

def base_dirs(L: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "T": L[2] - L[1],
        "I": L[6] - L[5],
        "M": L[10] - L[9],
        "R": L[14] - L[13],
        "P": L[18] - L[17],
    }

def finger_joint_angles(L: Dict[int, np.ndarray], chain: List[int]) -> List[float]:
    def joint(a,b,c):
        v1 = L[a] - L[b]
        v2 = L[c] - L[b]
        return angle_between(v1, v2)
    return [joint(chain[i-1], chain[i], chain[i+1]) for i in range(1, len(chain)-1)]

def parametrise(L: Dict[int, np.ndarray]) -> Dict:
    n_p, _ = compute_palm_plane(L)
    n_t, _ = compute_thumb_plane(L)
    phi_thumb = angle_between(n_p, n_t)

    dirs = base_dirs(L)
    inter = {
        "T_I": angle_between(dirs["T"], dirs["I"]),
        "I_M": angle_between(dirs["I"], dirs["M"]),
        "M_R": angle_between(dirs["M"], dirs["R"]),
        "R_P": angle_between(dirs["R"], dirs["P"]),
    }
    joints = {
        "T": finger_joint_angles(L, FINGER_CHAINS["T"]),
        "I": finger_joint_angles(L, FINGER_CHAINS["I"]),
        "M": finger_joint_angles(L, FINGER_CHAINS["M"]),
        "R": finger_joint_angles(L, FINGER_CHAINS["R"]),
        "P": finger_joint_angles(L, FINGER_CHAINS["P"]),
    }
    return {"phi_thumb": phi_thumb, "inter": inter, "joints": joints}

# --------------------- Invariance-aware normalisation ---------------------

def normalise_landmarks(L: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Apply invariances:
    - Translation: center by wrist (0) or palm center.
    - Scale: make distance between index MCP (5) and pinky MCP (17) equal to 1.
    - Rotation: align palm plane normal with +Z; then normalize in-plane rotation to [-90°,90°]
      by aligning vector (index MCP -> pinky MCP) to +X as closely as possible, picking the
      rotation with |angle| ≤ 90°.
    """
    L = {i: L[i].astype(float).copy() for i in L}

    # 1) translation
    center = 0.25*(L[5] + L[9] + L[13] + L[17])  # palm center via MCPs
    L = translate(L, -center)

    # 2) scale
    width = np.linalg.norm((L[17] - L[5]))
    if width < 1e-6:
        width = 1.0
    L = scale(L, 1.0 / width)

    # 3) rotate palm plane normal to +Z
    n_p, _ = compute_palm_plane(L)
    R1 = rotation_to_align(n_p, np.array([0.0, 0.0, 1.0]))
    L = apply_rotation(L, R1)

    # 4) in-plane rotation normalization (about Z)
    # align vector 5->17 (index MCP to pinky MCP) as close as possible to +X within ±90°
    v = L[17] - L[5]
    v_xy = v.copy(); v_xy[2] = 0.0
    ang = math.atan2(v_xy[1], v_xy[0])  # angle from +X
    # Bring angle into [-pi/2, pi/2] by rotating +/- pi if needed
    if ang > math.pi/2:
        ang -= math.pi
    elif ang < -math.pi/2:
        ang += math.pi
    c, s = math.cos(-ang), math.sin(-ang)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    L = apply_rotation(L, Rz)

    return L

# --------------------- Synthetic hand from parameters (illustrative) ---------------------

def synthesize_from_params(params: Dict) -> Dict[int, np.ndarray]:
    """
    Build a simple synthetic hand consistent with a parameter vector.
    This is not a biomechanical model; it's illustrative for visualisation.
    Assumes palm on z=0 plane and wrist near origin.
    """
    # Base MCP anchors in a canonical palm frame
    L = {}
    L[0] = np.array([0.0, -0.4, 0.0])  # wrist
    mcp_x = [-0.7, -0.25, 0.0, 0.25, 0.55]  # thumb CMC, index MCP, middle MCP, ring MCP, pinky MCP
    mcp_y = [ -0.05, 0.0, 0.0, 0.0, 0.0]
    ids = [1,5,9,13,17]
    for i,(x,y) in enumerate(zip(mcp_x, mcp_y)):
        L[ids[i]] = np.array([x, y, 0.0])

    # Determine base directions for fingers from inter-finger angles
    # Start with index direction roughly +Y, and build others by rotating in-plane
    base_dir_I = unit(np.array([0.0, 1.0, 0.0]))
    theta_TI = np.radians(params["inter"]["T_I"])
    theta_IM = np.radians(params["inter"]["I_M"])
    theta_MR = np.radians(params["inter"]["M_R"])
    theta_RP = np.radians(params["inter"]["R_P"])

    def rotz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    dirs = {}
    dirs["I"] = base_dir_I
    dirs["M"] = unit(rotz(theta_IM) @ dirs["I"])
    dirs["R"] = unit(rotz(theta_IM + theta_MR) @ dirs["I"])
    dirs["P"] = unit(rotz(theta_IM + theta_MR + theta_RP) @ dirs["I"])
    # thumb direction opposite side
    dirs["T"] = unit(rotz(-theta_TI) @ dirs["I"])

    # Thumb base plane tilt vs palm (phi_thumb)
    phi_t = np.radians(params["phi_thumb"])
    thumb_dir_3d = unit(np.array([dirs["T"][0], dirs["T"][1], np.tan(phi_t)]))

    # Segment lengths (approx)
    segs = {
        "T": [0.45, 0.35, 0.25],
        "I": [0.55, 0.45, 0.30],
        "M": [0.60, 0.48, 0.31],
        "R": [0.58, 0.45, 0.30],
        "P": [0.52, 0.40, 0.28],
    }

    # Build chains with per-joint angles in their (local) finger plane
    def build_chain(start_id, base_dir, seg_lengths, joint_angles_deg, vertical_lift=0.0):
        pts = []
        P = L[start_id]
        # Choose finger plane as spanned by base_dir and z
        z = np.array([0.0, 0.0, 1.0])
        plane_u = unit(base_dir)
        plane_v = unit(np.cross(z, plane_u))
        if np.linalg.norm(plane_v) < 1e-6:
            plane_v = np.array([1.0, 0.0, 0.0])
        # First segment
        P1 = P + seg_lengths[0]*plane_u + vertical_lift*z*0.0
        pts.append(P1)
        angs = joint_angles_deg
        # Next segments rotate by joint angles around plane normal
        cur_dir = plane_u
        cur_pt = P1
        for k in range(1, len(seg_lengths)):
            # rotate cur_dir towards -plane_u by angle angs[k-1] within the plane (flexion)
            # model: cur_dir_new = rot_in_plane(cur_dir, +/- angle)
            sign = -1.0
            a = np.radians(angs[k-1])
            # rotate in the plane using z-normal (approx)
            Rz = rotz(sign*a)
            new_dir = unit(Rz @ cur_dir)
            nxt = cur_pt + seg_lengths[k]*new_dir
            pts.append(nxt)
            cur_dir = new_dir
            cur_pt = nxt
        return pts

    # Thumb
    L[2], L[3], L[4] = build_chain(1, thumb_dir_3d, segs["T"], params["joints"]["T"])
    # Other fingers
    for key, start, seg in [("I",5,segs["I"]),("M",9,segs["M"]),("R",13,segs["R"]),("P",17,segs["P"])]:
        p1,p2,p3 = build_chain(start, dirs[key], seg, params["joints"][key])
        chain = FINGER_CHAINS[key]
        L[chain[1]] = p1; L[chain[2]] = p2; L[chain[3]] = p3

    return L

# --------------------- Plotting ---------------------

def plot_hand_3d(L: Dict[int,np.ndarray], setup: Setup, show_cone=True, title="") -> go.Figure:
    xs, ys, zs = [], [], []
    segs_x, segs_y, segs_z = [], [], []
    for (i,j) in CONNECTIONS:
        if i in L and j in L:
            xs += [L[i][0], L[j][0], None]
            ys += [L[i][1], L[j][1], None]
            zs += [L[i][2], L[j][2], None]
    pts_x = [L[i][0] for i in L]
    pts_y = [L[i][1] for i in L]
    pts_z = [L[i][2] for i in L]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, 
        mode="lines", 
        name="Hand skeleton",
        line=dict(color="#1f77b4", width=4),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_x, y=pts_y, z=pts_z, 
        mode="markers", 
        name="Landmarks",
        marker=dict(size=5, color="#ff7f0e", line=dict(color='white', width=1)),
        hovertemplate='Point %{text}<extra></extra>',
        text=list(L.keys())
    ))

    # Torch
    fig.add_trace(go.Scatter3d(
        x=[setup.T[0]], y=[setup.T[1]], z=[setup.T[2]],
        mode="markers+text", 
        text=["💡"], 
        name="Torch",
        marker=dict(size=10, color='#ffd700', symbol='diamond'),
        textfont=dict(size=14)
    ))

    # Wall circle base
    n = setup.wall_n
    # wall center (closest point to origin along n scaled by d)
    center = n * (setup.wall_d/np.dot(n,n))
    # basis on wall
    u = unit(np.cross(n, np.array([1.0,0.0,0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(n, np.array([0.0,1.0,0.0])))
    v = unit(np.cross(n, u))
    thetas = np.linspace(0, 2*np.pi, 120)
    circle = np.array([center + setup.base_radius*(np.cos(t)*u + np.sin(t)*v) for t in thetas])
    fig.add_trace(go.Scatter3d(
        x=circle[:,0], y=circle[:,1], z=circle[:,2], 
        mode="lines", 
        name="Wall base",
        line=dict(color='#2ca02c', width=3),
        hoverinfo='skip'
    ))

    if show_cone:
        # draw a few generators
        for k in range(0, len(thetas), 15):
            P = circle[k]
            fig.add_trace(go.Scatter3d(
                x=[setup.T[0], P[0]], y=[setup.T[1], P[1]], z=[setup.T[2], P[2]],
                mode="lines", 
                line=dict(width=1, color='rgba(211, 211, 211, 0.3)', dash='dash'), 
                name="Projection ray", 
                showlegend=(k==0),
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1f77b4')),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title='X', gridcolor='lightgray', showbackground=True, backgroundcolor='rgba(240,240,240,0.3)'),
            yaxis=dict(title='Y', gridcolor='lightgray', showbackground=True, backgroundcolor='rgba(240,240,240,0.3)'),
            zaxis=dict(title='Z', gridcolor='lightgray', showbackground=True, backgroundcolor='rgba(240,240,240,0.3)')
        ),
        height=550, 
        margin=dict(l=0,r=0,t=50,b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

def plot_projection_2d(L: Dict[int,np.ndarray], setup: Setup) -> go.Figure:
    # project and plot in 2D on wall basis
    n = setup.wall_n
    u = unit(np.cross(n, np.array([1.0,0.0,0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(n, np.array([0.0,1.0,0.0])))
    v = unit(np.cross(n, u))
    def to2d(X):
        Y = project_point_to_plane_from_T(X, setup)
        return np.array([np.dot(Y, u), np.dot(Y, v)])
    pts = {i: to2d(L[i]) for i in L}
    xs, ys = [], []
    for (i,j) in CONNECTIONS:
        if i in pts and j in pts:
            xs += [pts[i][0], pts[j][0], None]
            ys += [pts[i][1], pts[j][1], None]
    dotx = [pts[i][0] for i in pts]
    doty = [pts[i][1] for i in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, 
        mode="lines", 
        name="Shadow outline",
        line=dict(color='#333333', width=3),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=dotx, y=doty, 
        mode="markers", 
        name="Shadow points",
        marker=dict(size=6, color='#d62728', line=dict(color='white', width=1)),
        hovertemplate='Point %{text}<extra></extra>',
        text=list(pts.keys())
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=dict(text="Shadow Projection on Wall", font=dict(size=18, color='#1f77b4')),
        height=500, 
        margin=dict(l=0,r=0,t=50,b=0),
        paper_bgcolor='#f5f5f5',
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray', title='Wall X'),
        yaxis=dict(gridcolor='lightgray', title='Wall Y'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

# --------------------- MediaPipe Webcam ---------------------

def mediapipe_frame_processor_factory(state_dict):
    import mediapipe as mp
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    drawing = mp.solutions.drawing_utils

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        h, w, _ = img.shape
        if res.multi_hand_landmarks:
            hand_landmarks = res.multi_hand_landmarks[0]
            # Convert normalized landmarks to a 3D numpy dict in a camera-centric frame
            L = {}
            for i, lm in enumerate(hand_landmarks.landmark):
                # x,y in [0,1]; we re-center to [-0.5,0.5], flip y to up.
                x = (lm.x - 0.5)
                y = -(lm.y - 0.5)
                z = -lm.z  # MediaPipe z is negative into the screen; flip for RHS
                L[i] = np.array([x, y, z], dtype=float)

            # Apply invariance normalisation
            L_norm = normalise_landmarks(L)
            # Save the latest sample in the shared state
            state_dict["latest_landmarks"] = {i: L_norm[i].tolist() for i in L_norm}
            state_dict["latest_params"] = parametrise(L_norm)

            # Draw 2D overlay for feedback
            drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback

# --------------------- UI ---------------------

def slider_degrees(label, value, minv=0.0, maxv=180.0, step=1.0):
    return st.slider(label, min_value=float(minv), max_value=float(maxv), value=float(value), step=float(step))

def parameters_editor(params: Dict) -> Dict:
    st.markdown("### 🎛️ Hand Parameter Editor")
    st.markdown("Adjust the sliders below to modify the hand pose and shape")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👐 Global & Inter-finger Angles")
        phi = slider_degrees("👍 Palm–thumb plane angle φ_thumb (deg)", params["phi_thumb"], 0.0, 120.0, 1.0)
        st.caption("Controls how far the thumb extends from the palm plane")
        
        st.markdown("**Spacing between fingers:**")
        i1 = slider_degrees("👍↔️☝️ Thumb to Index", params["inter"]["T_I"], 5.0, 120.0, 1.0)
        i2 = slider_degrees("☝️↔️🖕 Index to Middle", params["inter"]["I_M"], 5.0, 120.0, 1.0)
        i3 = slider_degrees("🖕↔️💍 Middle to Ring", params["inter"]["M_R"], 5.0, 120.0, 1.0)
        i4 = slider_degrees("💍↔️🤙 Ring to Pinky", params["inter"]["R_P"], 5.0, 120.0, 1.0)

    with col2:
        st.markdown("#### 🖐️ Individual Finger Joint Angles")
        st.caption("Control the bend at each joint (higher = more bent)")
        
        def edit_finger(name, emoji, vals):
            st.markdown(f"**{emoji} {name}:**")
            a = slider_degrees(f"{name} MCP (base)", vals[0], 0.0, 180.0, 1.0)
            b = slider_degrees(f"{name} PIP (middle)", vals[1], 0.0, 180.0, 1.0)
            c = slider_degrees(f"{name} DIP (tip)", vals[2] if len(vals)>2 else 10.0, 0.0, 180.0, 1.0)
            return [a,b,c]
        jT = edit_finger("Thumb", "👍", params["joints"]["T"] + [10.0])
        jI = edit_finger("Index", "☝️", params["joints"]["I"])
        jM = edit_finger("Middle", "🖕", params["joints"]["M"])
        jR = edit_finger("Ring", "💍", params["joints"]["R"])
        jP = edit_finger("Pinky", "🤙", params["joints"]["P"])

    newp = {
        "phi_thumb": float(phi),
        "inter": {"T_I": float(i1), "I_M": float(i2), "M_R": float(i3), "R_P": float(i4)},
        "joints": {"T": jT[:3], "I": jI[:3], "M": jM[:3], "R": jR[:3], "P": jP[:3]},
    }
    return newp

def default_params() -> Dict:
    return {
        "phi_thumb": 25.0,
        "inter": {"T_I": 35.0, "I_M": 18.0, "M_R": 15.0, "R_P": 18.0},
        "joints": {
            "T": [20.0, 25.0, 20.0],
            "I": [10.0, 15.0, 10.0],
            "M": [8.0, 12.0, 10.0],
            "R": [10.0, 14.0, 10.0],
            "P": [12.0, 16.0, 12.0],
        }
    }

def dict_to_np(d):
    return {int(k): np.array(v, dtype=float) for k,v in d.items()}

def main():
    # Apply custom styling
    apply_custom_css()
    
    # Sidebar with information and settings
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/google/mediapipe/master/docs/images/mobile/hand_landmarks.png", 
                 use_container_width=True, caption="MediaPipe Hand Landmarks")
        
        st.markdown("## 📚 About")
        st.markdown("""
        This interactive app demonstrates hand gesture capture and parametrization:
        
        **Features:**
        - 🎥 Real-time webcam capture
        - 📊 Parametric representation
        - 🎨 3D visualization
        - 🌑 Shadow projection
        - ✏️ Interactive editing
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Technical Info")
        st.info("""
        **Invariances Applied:**
        - Translation (palm-centered)
        - Scale (normalized)
        - Rotation (aligned to ±90°)
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        1. Allow camera access when prompted
        2. Show your palm clearly to the camera
        3. Freeze a pose to edit parameters
        4. Experiment with different angles
        5. View shadow projections in Tab 3
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io) & [MediaPipe](https://mediapipe.dev)")
    
    # Main content
    st.title("🖐️ Shadow Puppet Parametrisation Viewer")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='font-size: 1.1em; margin: 0;'>
    This app <b>captures hand gestures</b> from your webcam using MediaPipe, extracts a 
    <b>parametric representation</b> with geometric invariances, and lets you 
    <b>visualize and edit</b> the hand pose in 3D space with shadow projections.
    </p>
    </div>
    """, unsafe_allow_html=True)

    setup = DEFAULT_SETUP

    # State for latest capture
    if "latest_landmarks" not in st.session_state:
        st.session_state["latest_landmarks"] = None
    if "latest_params" not in st.session_state:
        st.session_state["latest_params"] = default_params()

    tabs = st.tabs(["📹 Live Capture", "✏️ Parameter Editor", "🌑 Shadow Projection"])

    # Tab 1: Live capture
    with tabs[0]:
        st.markdown("### 📹 Real-time Hand Capture")
        st.markdown("""
        **Instructions:**
        1. Allow camera access when prompted
        2. Position your hand in the camera frame
        3. MediaPipe will detect and track 21 landmarks
        4. View extracted parameters in real-time
        5. Click "Freeze to Editor" to save current pose
        """)
        
        st.info("💡 **Tip:** Keep your hand clearly visible with good lighting for best results!")
        
        state_dict = st.session_state  # pass reference to callback
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="hand-capture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=mediapipe_frame_processor_factory(state_dict),
        )

        st.markdown("---")
        
        colA, colB = st.columns([1,1])
        with colA:
            st.markdown("#### 📊 Extracted Parameters")
            if st.session_state.get("latest_params"):
                st.json(st.session_state["latest_params"])
            else:
                st.warning("⚠️ No hand detected yet. Show your hand to the camera!")
        
        with colB:
            st.markdown("#### 🎨 3D Visualization")
            if st.session_state["latest_landmarks"]:
                L = dict_to_np(st.session_state["latest_landmarks"])
                fig3d = plot_hand_3d(L, setup, title="Normalized Captured Hand")
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("📸 3D visualization will appear here once a hand is detected")

        st.markdown("---")
        st.markdown("### 🔒 Freeze Current Pose")
        st.markdown("Save the current hand parameters to edit them in the Parameter Editor tab")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
        with col_btn2:
            if st.button("🔒 Freeze to Editor", use_container_width=True):
                if st.session_state["latest_params"]:
                    st.session_state["frozen_params"] = st.session_state["latest_params"]
                    st.success("✅ Parameters frozen! Switch to the 'Parameter Editor' tab to modify them.")
                else:
                    st.error("❌ No parameters to freeze. Capture a hand first!")

    # Tab 2: Editor & synthesis
    with tabs[1]:
        st.markdown("### ✏️ Parameter Editor & Synthesis")
        st.markdown("""
        Adjust hand parameters using the sliders below and see the synthetic hand update in real-time.
        Start with frozen parameters from captured poses or use default values.
        """)
        
        # Show source of parameters
        if "frozen_params" in st.session_state:
            st.success("📌 Using frozen parameters from captured pose")
        else:
            st.info("ℹ️ Using default or latest captured parameters")
        
        st.markdown("---")
        
        base_params = st.session_state.get("frozen_params", st.session_state["latest_params"] or default_params())
        edited = parameters_editor(base_params)
        st.session_state["edited_params"] = edited

        st.markdown("---")
        st.markdown("### 🎭 Synthetic Hand Preview")
        st.markdown("The 3D hand below is reconstructed from the parameters you set above")
        
        L_syn = synthesize_from_params(edited)
        st.plotly_chart(plot_hand_3d(L_syn, setup, title="Synthesized Hand from Parameters"), use_container_width=True)
        
        # Option to reset parameters
        col_reset1, col_reset2, col_reset3 = st.columns([1,2,1])
        with col_reset2:
            if st.button("🔄 Reset to Default Parameters", use_container_width=True):
                if "frozen_params" in st.session_state:
                    del st.session_state["frozen_params"]
                st.rerun()

    # Tab 3: Projection
    with tabs[2]:
        st.markdown("### 🌑 Shadow Projection Simulator")
        st.markdown("""
        This tab simulates casting a shadow of the hand onto a virtual wall using a torch as the light source.
        The projection geometry includes:
        - 💡 **Torch**: Light source position
        - 🟢 **Wall**: Circular projection surface
        - 📏 **Projection rays**: Lines from torch through hand to wall
        """)
        
        st.markdown("---")
        
        col_mode1, col_mode2 = st.columns([1,3])
        with col_mode1:
            st.markdown("**Select hand source:**")
        with col_mode2:
            mode = st.radio(
                "Choose source",
                ["📹 Captured (normalized)", "✏️ Synthesized from editor"],
                label_visibility="collapsed"
            )
        
        if mode.startswith("📹") and st.session_state["latest_landmarks"]:
            L = dict_to_np(st.session_state["latest_landmarks"])
            st.success("Using captured hand from webcam")
        else:
            L = synthesize_from_params(st.session_state.get("edited_params", default_params()))
            st.info("Using synthesized hand from parameter editor")
        
        st.markdown("---")
        
        col_3d, col_2d = st.columns([1,1])
        
        with col_3d:
            st.markdown("#### 🎨 3D Scene with Projection Geometry")
            st.plotly_chart(plot_hand_3d(L, setup, title="3D Hand with Torch & Projection Cone"), use_container_width=True)
        
        with col_2d:
            st.markdown("#### 🌑 Shadow on Wall (2D)")
            st.plotly_chart(plot_projection_2d(L, setup), use_container_width=True)

    with st.expander("📖 Technical Details & Advanced Information", expanded=False):
        st.markdown("## 🔬 Technical Documentation")
        
        st.markdown("### 🎯 Geometric Invariances")
        st.markdown("""
        The app applies three key invariances to normalize hand poses for consistent comparison:
        
        1. **Translation Invariance** 🔄
           - Centers the hand at the palm (metacarpophalangeal joint center)
           - Removes dependency on hand position in space
        
        2. **Scale Invariance** 📏
           - Normalizes to unit distance between index and pinky MCP joints
           - Makes parameters independent of hand size
        
        3. **Rotation Invariance** 🔃
           - Aligns palm plane normal to +Z axis
           - Wraps in-plane rotation to ±90° range
           - Aligns index→pinky direction with +X axis
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Hand Parametrization")
        st.markdown("""
        The parametric representation captures hand configuration with:
        
        **Global Parameters:**
        - **φ_thumb**: Palm-thumb plane angle (controls thumb extension)
        
        **Inter-finger Angles:**
        - Thumb to Index (T-I)
        - Index to Middle (I-M)
        - Middle to Ring (M-R)
        - Ring to Pinky (R-P)
        
        **Joint Angles (per finger):**
        - MCP: Metacarpophalangeal (base) joint
        - PIP: Proximal interphalangeal (middle) joint
        - DIP: Distal interphalangeal (tip) joint
        
        Total: **1 + 4 + (5×3) = 20 parameters** describe the hand pose
        """)
        
        st.markdown("---")
        
        st.markdown("### 🤖 MediaPipe Integration")
        st.markdown("""
        MediaPipe provides 21 3D landmarks:
        - 1 wrist point
        - 4 thumb points
        - 4×4 finger points (index, middle, ring, pinky)
        
        Coordinates are normalized:
        - x, y ∈ [0, 1] (image space)
        - z relative depth
        
        We map these to a canonical 3D frame before parametrization.
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎭 Synthesis Process")
        st.warning("""
        **Important Note:** The synthesis step is **illustrative** and educational, not a full biomechanical model.
        It demonstrates how parameters affect hand configuration but doesn't enforce true anatomical constraints.
        Use it to explore parameter effects and understand the representation.
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - 🌞 Use good lighting for camera capture
        - 🤲 Keep entire hand visible in frame
        - 📐 Hold hand steady for stable parameters
        - 🎨 Try extreme parameter values to see effects
        - 🔄 Freeze interesting poses to study them
        - 🌑 View projections to understand 3D structure
        """)

if __name__ == "__main__":
    main()

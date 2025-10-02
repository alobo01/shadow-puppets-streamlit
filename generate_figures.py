import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

# MediaPipe hand connections (subset for clarity)
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),           # thumb chain
    (0,5), (5,6), (6,7), (7,8),           # index chain
    (0,9), (9,10), (10,11), (11,12),      # middle chain
    (0,13), (13,14), (14,15), (15,16),    # ring chain
    (0,17), (17,18), (18,19), (19,20),    # pinky chain
    (5,9), (9,13), (13,17)                # palm links
]

FINGER_CHAINS = {
    "T": [1,2,3,4],        # thumb: CMC->MCP->IP->TIP
    "I": [5,6,7,8],        # index: MCP->PIP->DIP->TIP
    "M": [9,10,11,12],     # middle
    "R": [13,14,15,16],    # ring
    "P": [17,18,19,20],    # pinky
}

@dataclass
class Setup:
    T: np.ndarray            # torch, shape (3,)
    wall_n: np.ndarray       # unit normal of wall plane
    wall_d: float            # plane offset (n^T x = d)
    base_radius: float       # cone base radius on wall

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0: 
        return v
    return v / n

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_u, b_u = unit(a), unit(b)
    val = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def plane_from_points(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
    n = unit(np.cross(q - p, r - p))
    d = float(np.dot(n, p))
    return n, d

def project_point_to_plane_from_T(X: np.ndarray, setup: Setup) -> np.ndarray:
    num = (setup.wall_d - np.dot(setup.wall_n, setup.T))
    den = np.dot(setup.wall_n, X - setup.T)
    if abs(den) < 1e-9:
        return X.copy()
    alpha = num / den
    return setup.T + alpha * (X - setup.T)

def generate_synthetic_hand(spread: float = 0.8, curl: float = 0.2) -> Dict[int, np.ndarray]:
    """
    Construct a plausible open-hand pose in 3D, centred at the origin.
    spread controls lateral finger splay; curl controls global finger bend.
    """
    L = {}
    # Wrist
    L[0] = np.array([0.0, 0.0, 0.0])
    # MCP x-positions for five digits (thumb to pinky)
    xs = np.array([-1.4, -0.5, 0.0, 0.5, 1.0]) * spread
    # y-depths for MCP row
    y0 = 1.2
    # typical segment lengths per finger (proximal, intermediate, distal)
    segs = {
        "T": [0.9, 0.7, 0.5],
        "I": [1.4, 1.1, 0.7],
        "M": [1.6, 1.2, 0.75],
        "R": [1.5, 1.1, 0.7],
        "P": [1.2, 0.9, 0.6],
    }
    # Base directions
    dirs = {
        "T": unit(np.array([-0.5,  0.8,  0.15])),
        "I": unit(np.array([-0.1,  1.0,  0.2])),
        "M": unit(np.array([ 0.0,  1.0,  0.1])),
        "R": unit(np.array([ 0.1,  1.0,  0.1])),
        "P": unit(np.array([ 0.25, 1.0,  0.15])),
    }
    # Landmarks along each finger
    # Thumb CMC at xs[0], MCP at a small offset along dir
    L[1] = np.array([xs[0], y0-0.4, 0.05])
    L[2] = L[1] + dirs["T"] * (0.5 + 0.2*curl)
    L[3] = L[2] + dirs["T"] * (segs["T"][1] * (1.0 - 0.3*curl))
    L[4] = L[3] + dirs["T"] * (segs["T"][2] * (1.0 - 0.5*curl))

    # Fingers I,M,R,P MCP positions
    mcp_ids = [5,9,13,17]
    for k, finger in enumerate(["I","M","R","P"]):
        mcp = np.array([xs[k+1], y0, 0.0])
        L[mcp_ids[k]] = mcp
        d = dirs[finger]
        # next joints
        L[mcp_ids[k]+1] = mcp + d * (segs[finger][0] * (1.0 - 0.2*curl))
        L[mcp_ids[k]+2] = L[mcp_ids[k]+1] + d * (segs[finger][1] * (1.0 - 0.4*curl))
        L[mcp_ids[k]+3] = L[mcp_ids[k]+2] + d * (segs[finger][2] * (1.0 - 0.6*curl))
    return L

def draw_hand_3d(ax, L: Dict[int, np.ndarray], annotate=False, title=None):
    for (i,j) in CONNECTIONS:
        if i in L and j in L:
            P, Q = L[i], L[j]
            ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]])
    xs = [L[i][0] for i in L]
    ys = [L[i][1] for i in L]
    zs = [L[i][2] for i in L]
    ax.scatter(xs, ys, zs, s=20)
    if annotate:
        for i in L:
            ax.text(L[i][0], L[i][1], L[i][2], str(i))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title: ax.set_title(title)
    ax.view_init(elev=20, azim=-70)
    set_equal_3d(ax)

def set_equal_3d(ax):
    # equal aspect helper
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5*max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits); y_middle = np.mean(y_limits); z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle-plot_radius, x_middle+plot_radius])
    ax.set_ylim3d([y_middle-plot_radius, y_middle+plot_radius])
    ax.set_zlim3d([z_middle-plot_radius, z_middle+plot_radius])

def draw_plane(ax, n: np.ndarray, d: float, center: np.ndarray, size: float = 2.5, alpha=0.2, label=None):
    # create a square patch on plane around center
    # find orthonormal basis u,v in the plane
    n = unit(n)
    u = unit(np.cross(n, np.array([1.0, 0.0, 0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(n, np.array([0.0, 1.0, 0.0])))
    v = unit(np.cross(n, u))
    # square corners
    corners = []
    for sx in [-1,1]:
        for sy in [-1,1]:
            corners.append(center + size*(sx*u + sy*v))
    # draw patch (two triangles)
    X = np.array([c[0] for c in corners] + [corners[0][0]])
    Y = np.array([c[1] for c in corners] + [corners[0][1]])
    Z = np.array([c[2] for c in corners] + [corners[0][2]])
    ax.plot(X[[0,1,3,2,0]], Y[[0,1,3,2,0]], Z[[0,1,3,2,0]])
    ax.plot([center[0]],[center[1]],[center[2]], marker='o')
    if label:
        ax.text(center[0],center[1],center[2],label)

def draw_cone(ax, setup: Setup, n_theta=40, n_h=2):
    # Draw the circle base on the wall and side generators to the torch
    # base circle center
    center = setup.wall_n * (setup.wall_d/np.linalg.norm(setup.wall_n)**2)
    # basis on the wall
    u = unit(np.cross(setup.wall_n, np.array([1.0, 0.0, 0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(setup.wall_n, np.array([0.0, 1.0, 0.0])))
    v = unit(np.cross(setup.wall_n, u))
    thetas = np.linspace(0, 2*math.pi, n_theta)
    circle = np.array([center + setup.base_radius*(math.cos(t)*u + math.sin(t)*v) for t in thetas])
    ax.plot(circle[:,0], circle[:,1], circle[:,2])
    # cone generators
    for k in range(0, n_theta, max(1,n_theta//12)):
        P = circle[k]
        ax.plot([setup.T[0], P[0]], [setup.T[1], P[1]], [setup.T[2], P[2]])
    # draw torch and normal
    ax.scatter([setup.T[0]], [setup.T[1]], [setup.T[2]], marker='^', s=60)
    ax.text(setup.T[0], setup.T[1], setup.T[2], "T")

def compute_palm_plane(L: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
    return plane_from_points(L[0], L[5], L[17])

def compute_thumb_plane(L: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
    C = 0.5*(L[9] + L[13])
    n, d = plane_from_points(L[1], L[2], C)
    return n, d

def base_dirs(L: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "T": L[2] - L[1],
        "I": L[6] - L[5],
        "M": L[10] - L[9],
        "R": L[14] - L[13],
        "P": L[18] - L[17],
    }

def plot_fig_cone_and_wall(L: Dict[int, np.ndarray], setup: Setup):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_cone(ax, setup)
    draw_hand_3d(ax, L, annotate=True, title="Torch, Wall, Projection Cone, and Hand")
    center = setup.wall_n * (setup.wall_d/np.linalg.norm(setup.wall_n)**2)
    draw_plane(ax, setup.wall_n, setup.wall_d, center, size=2.5, label="Wall")
    fig.savefig(FIGDIR / "fig_cone_wall.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def plot_fig_palm_thumb_planes(L: Dict[int, np.ndarray]):
    n_p, d_p = compute_palm_plane(L)
    n_t, d_t = compute_thumb_plane(L)
    center_p = (d_p * n_p)
    center_t = (d_t * n_t)
    angle = angle_between(n_p, n_t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_hand_3d(ax, L, annotate=False, title=f"Palm vs Thumb planes (angle ≈ {angle:.1f}°)")
    draw_plane(ax, n_p, d_p, center_p, size=2.2, label="Palm")
    draw_plane(ax, n_t, d_t, center_t, size=1.8, label="Thumb")
    fig.savefig(FIGDIR / "fig_palm_thumb_planes.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def plot_fig_interfinger_angles(L: Dict[int, np.ndarray]):
    dirs = base_dirs(L)
    pairs = [("T","I"), ("I","M"), ("M","R"), ("R","P")]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_hand_3d(ax, L, annotate=True, title="Inter-finger angles at the base")
    # draw short rays from each finger's first point
    anchors = {"T": L[1], "I": L[5], "M": L[9], "R": L[13], "P": L[17]}
    for key in anchors:
        P = anchors[key]
        D = unit(dirs[key])
        Q = P + 1.0 * D
        ax.plot([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]])
        ax.text(Q[0], Q[1], Q[2], key)
    # angle annotations
    text = []
    for (a,b) in pairs:
        ang = angle_between(dirs[a], dirs[b])
        text.append(f"{a}–{b}: {ang:.1f}°")
    ax.text2D(0.02, 0.95, "\n".join(text), transform=ax.transAxes)
    fig.savefig(FIGDIR / "fig_interfinger_angles.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def finger_joint_angles(L: Dict[int, np.ndarray], chain: List[int]) -> List[float]:
    def joint(a,b,c):
        v1 = L[a] - L[b]
        v2 = L[c] - L[b]
        return angle_between(v1, v2)
    # joints along the chain (internal vertices)
    return [joint(chain[i-1], chain[i], chain[i+1]) for i in range(1, len(chain)-1)]

def plot_fig_joint_angles_for_index(L: Dict[int, np.ndarray]):
    chain = FINGER_CHAINS["I"]
    angles = finger_joint_angles(L, chain)
    # Draw the index finger chain in its own best-fit plane (by PCA)
    P = np.stack([L[i] for i in chain], axis=0)
    # PCA for plane basis
    C = P.mean(axis=0)
    X = P - C
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:2,:]  # two principal directions
    coords = X @ basis.T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(coords[:,0], coords[:,1], marker='o')
    for i, ang in enumerate(angles, start=1):
        ax.text(coords[i,0], coords[i,1], f"{ang:.1f}°")
    ax.set_aspect('equal', 'box')
    ax.set_title("Index finger joint angles (in finger plane)")
    fig.savefig(FIGDIR / "fig_joint_angles_index.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def plot_fig_projection(L: Dict[int, np.ndarray], setup: Setup):
    # project all landmarks to wall
    P2 = {i: project_point_to_plane_from_T(L[i], setup) for i in L}
    # choose an orthonormal basis on the wall to render 2D
    n = setup.wall_n
    u = unit(np.cross(n, np.array([1.0, 0.0, 0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(n, np.array([0.0, 1.0, 0.0])))
    v = unit(np.cross(n, u))
    def to2d(X):
        Y = X  # already on the plane
        return np.array([np.dot(Y, u), np.dot(Y, v)])
    pts2d = {i: to2d(P2[i]) for i in P2}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (i,j) in CONNECTIONS:
        if i in pts2d and j in pts2d:
            P, Q = pts2d[i], pts2d[j]
            ax.plot([P[0],Q[0]], [P[1],Q[1]])
    X = np.array([pts2d[i][0] for i in pts2d])
    Y = np.array([pts2d[i][1] for i in pts2d])
    ax.scatter(X, Y, s=20)
    ax.set_aspect('equal', 'box')
    ax.set_title("Projected skeleton on the wall")
    fig.savefig(FIGDIR / "fig_projection.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def plot_fig_family_variation(setup: Setup):
    # Sweep one parameter: inter-finger I–M angle by rotating middle base direction slightly
    L0 = generate_synthetic_hand(spread=0.85, curl=0.1)
    anchors = [9,10,11,12]  # middle chain
    origin = L0[9].copy()
    def rotz(angle_deg):
        t = np.radians(angle_deg)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        return R
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ang in [-10, 0, 10, 20]:
        L = {i: L0[i].copy() for i in L0}
        R = rotz(ang)
        for idx in anchors:
            L[idx] = origin + R @ (L[idx]-origin)
        draw_hand_3d(ax, L, annotate=False, title="Family sweep example (I–M angle)")
    fig.savefig(FIGDIR / "fig_family_sweep.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

def main():
    # Scene setup: wall is z = 4; torch sits at z = -2 on axis
    setup = Setup(
        T=np.array([0.0, 0.0, -2.0]),
        wall_n=np.array([0.0, 0.0, 1.0]),
        wall_d=4.0,
        base_radius=2.0,
    )
    # Hand
    L = generate_synthetic_hand(spread=0.9, curl=0.15)

    # Produce figures (each in a separate plot, no subplots)
    plot_fig_cone_and_wall(L, setup)
    plot_fig_palm_thumb_planes(L)
    plot_fig_interfinger_angles(L)
    plot_fig_joint_angles_for_index(L)
    plot_fig_projection(L, setup)
    plot_fig_family_variation(setup)

if __name__ == "__main__":
    main()

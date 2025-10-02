"""
Improved Shadow Puppet Visualization Script

This script creates clear, illustrative visualizations for the shadow puppet parametrization,
showing all key concepts including:
- MediaPipe hand landmarks
- Torch/wall/projection cone setup
- Palm and thumb plane definitions
- Inter-finger angles
- Joint angles within finger planes
- Shadow projection process
- Parameter families and continuous variations

Author: Enhanced visualization for shadow puppet formalization
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, Wedge
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Set up high-quality plotting defaults
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

FIGDIR = Path(__file__).parent / "improved_figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

# Enhanced MediaPipe hand connections with anatomical grouping
FINGER_CONNECTIONS = {
    'thumb': [(0,1), (1,2), (2,3), (3,4)],
    'index': [(0,5), (5,6), (6,7), (7,8)],
    'middle': [(0,9), (9,10), (10,11), (11,12)],
    'ring': [(0,13), (13,14), (14,15), (15,16)],
    'pinky': [(0,17), (17,18), (18,19), (19,20)]
}

PALM_CONNECTIONS = [(5,9), (9,13), (13,17)]

ALL_CONNECTIONS = []
for finger_conns in FINGER_CONNECTIONS.values():
    ALL_CONNECTIONS.extend(finger_conns)
ALL_CONNECTIONS.extend(PALM_CONNECTIONS)

# Landmark names for annotation
LANDMARK_NAMES = {
    0: 'WRIST',
    1: 'T_CMC', 2: 'T_MCP', 3: 'T_IP', 4: 'T_TIP',
    5: 'I_MCP', 6: 'I_PIP', 7: 'I_DIP', 8: 'I_TIP',
    9: 'M_MCP', 10: 'M_PIP', 11: 'M_DIP', 12: 'M_TIP',
    13: 'R_MCP', 14: 'R_PIP', 15: 'R_DIP', 16: 'R_TIP',
    17: 'P_MCP', 18: 'P_PIP', 19: 'P_DIP', 20: 'P_TIP'
}

FINGER_CHAINS = {
    "T": [1,2,3,4],        # thumb: CMC->MCP->IP->TIP
    "I": [5,6,7,8],        # index: MCP->PIP->DIP->TIP
    "M": [9,10,11,12],     # middle
    "R": [13,14,15,16],    # ring
    "P": [17,18,19,20],    # pinky
}

# Color scheme for consistent visualization
COLORS = {
    'thumb': '#FF6B6B',
    'index': '#4ECDC4', 
    'middle': '#45B7D1',
    'ring': '#96CEB4',
    'pinky': '#FFEAA7',
    'palm': '#DDA0DD',
    'torch': '#FFD700',
    'wall': '#B0C4DE',
    'cone': '#FFA07A',
    'shadow': '#696969'
}

@dataclass
class Setup:
    T: np.ndarray            # torch position, shape (3,)
    wall_n: np.ndarray       # unit normal of wall plane
    wall_d: float            # plane offset (n^T x = d)
    base_radius: float       # cone base radius on wall

def unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees."""
    a_u, b_u = unit(a), unit(b)
    val = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def plane_from_points(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
    """Define plane from three points."""
    n = unit(np.cross(q - p, r - p))
    d = float(np.dot(n, p))
    return n, d

def project_point_to_plane_from_T(X: np.ndarray, setup: Setup) -> np.ndarray:
    """Project point X onto wall plane from torch T."""
    num = (setup.wall_d - np.dot(setup.wall_n, setup.T))
    den = np.dot(setup.wall_n, X - setup.T)
    if abs(den) < 1e-9:
        return X.copy()
    alpha = num / den
    return setup.T + alpha * (X - setup.T)

def generate_realistic_hand(params: Optional[Dict] = None) -> Dict[int, np.ndarray]:
    """
    Generate a realistic hand pose using the parametrization from sample_params.json
    """
    # Load default parameters
    default_params = {
        "phi_thumb": 25.0,
        "inter": {"T_I": 35.0, "I_M": 20.0, "M_R": 18.0, "R_P": 20.0},
        "joints": {
            "T": [25.0, 25.0, 20.0],
            "I": [12.0, 15.0, 12.0],
            "M": [10.0, 12.0, 10.0],
            "R": [12.0, 14.0, 12.0],
            "P": [14.0, 16.0, 14.0]
        }
    }
    
    if params:
        # Merge with provided parameters
        for key in params:
            if key in default_params:
                if isinstance(default_params[key], dict):
                    default_params[key].update(params[key])
                else:
                    default_params[key] = params[key]
    
    L = {}
    # Wrist at origin
    L[0] = np.array([0.0, 0.0, 0.0])
    
    # Define palm plane (simplified as xy-plane initially)
    palm_center = np.array([0.0, 1.2, 0.0])
    
    # MCP positions for fingers (spread based on inter-finger angles)
    base_spread = 0.8
    mcp_positions = {
        'I': np.array([-0.4 * base_spread, 1.3, 0.0]),
        'M': np.array([0.0, 1.4, 0.0]),
        'R': np.array([0.4 * base_spread, 1.3, 0.0]),
        'P': np.array([0.8 * base_spread, 1.1, 0.0])
    }
    
    # Thumb CMC position (offset from wrist)
    L[1] = np.array([-1.2 * base_spread, 0.8, 0.2])
    
    # Generate finger positions based on joint angles
    finger_lengths = {
        'T': [0.8, 0.6, 0.5],
        'I': [1.2, 0.9, 0.6],
        'M': [1.4, 1.0, 0.7],
        'R': [1.3, 0.9, 0.6],
        'P': [1.0, 0.8, 0.5]
    }
    
    # Build each finger
    for finger, chain in FINGER_CHAINS.items():
        if finger == 'T':
            # Thumb special case - starts from CMC
            start_pos = L[1]
            # Thumb direction (toward center of palm with thumb angle)
            thumb_dir = unit(np.array([0.7, 0.5, 0.3]))
        else:
            # Other fingers start from MCP
            finger_map = {'I': 'I', 'M': 'M', 'R': 'R', 'P': 'P'}
            start_pos = mcp_positions[finger_map[finger]]
            L[chain[0]] = start_pos
            # Base direction (roughly toward fingertips)
            base_dir = np.array([0.0, 1.0, 0.2])
            
        # Get joint angles for this finger
        joint_angles = default_params['joints'][finger]
        
        # Build finger segments
        current_pos = start_pos
        current_dir = thumb_dir if finger == 'T' else base_dir
        
        for i, (joint_id, length) in enumerate(zip(chain[1:], finger_lengths[finger])):
            # Apply joint bend (simplified)
            bend_factor = 1.0 - (joint_angles[i] / 180.0) * 0.3
            segment_dir = current_dir * bend_factor
            
            L[joint_id] = current_pos + segment_dir * length
            current_pos = L[joint_id]
    
    return L

def set_equal_3d_aspect(ax, padding=0.1):
    """Set equal aspect ratio for 3D plot with padding."""
    # Get the string of all the axis limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # The plot's bounding cube is a cube with sides of length max_range
    max_range = max([x_range, y_range, z_range]) * (1 + padding)
    
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def draw_coordinate_system(ax, origin=np.array([0,0,0]), scale=1.0, alpha=0.7):
    """Draw coordinate system axes."""
    axes_vectors = np.array([[scale,0,0], [0,scale,0], [0,0,scale]])
    axes_colors = ['red', 'green', 'blue']
    axes_labels = ['X', 'Y', 'Z']
    
    for i, (vec, color, label) in enumerate(zip(axes_vectors, axes_colors, axes_labels)):
        end_point = origin + vec
        ax.quiver(origin[0], origin[1], origin[2], 
                 vec[0], vec[1], vec[2], 
                 color=color, alpha=alpha, arrow_length_ratio=0.1)
        ax.text(end_point[0], end_point[1], end_point[2], 
               label, color=color, fontsize=8)

def draw_enhanced_hand_3d(ax, L: Dict[int, np.ndarray], 
                         annotate_landmarks=True, 
                         show_finger_colors=True,
                         title=None):
    """Draw hand with enhanced visualization including colors and proper annotations."""
    
    # Draw connections with finger-specific colors
    if show_finger_colors:
        for finger, connections in FINGER_CONNECTIONS.items():
            color = COLORS[finger]
            for (i,j) in connections:
                if i in L and j in L:
                    P, Q = L[i], L[j]
                    ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                           color=color, linewidth=2.5, alpha=0.8)
        
        # Draw palm connections
        for (i,j) in PALM_CONNECTIONS:
            if i in L and j in L:
                P, Q = L[i], L[j]
                ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                       color=COLORS['palm'], linewidth=2, alpha=0.7)
    else:
        # Draw all connections in single color
        for (i,j) in ALL_CONNECTIONS:
            if i in L and j in L:
                P, Q = L[i], L[j]
                ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                       'k-', linewidth=1.5, alpha=0.8)
    
    # Draw landmarks as colored spheres
    for i in L:
        # Determine color based on finger
        color = 'black'
        if i == 0:
            color = 'purple'  # wrist
        elif 1 <= i <= 4:
            color = COLORS['thumb']
        elif 5 <= i <= 8:
            color = COLORS['index']
        elif 9 <= i <= 12:
            color = COLORS['middle']
        elif 13 <= i <= 16:
            color = COLORS['ring']
        elif 17 <= i <= 20:
            color = COLORS['pinky']
            
        ax.scatter([L[i][0]], [L[i][1]], [L[i][2]], 
                  s=40, c=color, alpha=0.9, edgecolors='black', linewidth=0.5)
        
        # Annotate landmarks
        if annotate_landmarks:
            ax.text(L[i][0], L[i][1], L[i][2], f'{i}', 
                   fontsize=7, color='black', weight='bold')
    
    # Styling
    ax.set_xlabel("X (cm)", fontsize=10)
    ax.set_ylabel("Y (cm)", fontsize=10) 
    ax.set_zlabel("Z (cm)", fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=12, weight='bold')
    
    # Better viewing angle
    ax.view_init(elev=15, azim=-60)
    set_equal_3d_aspect(ax)
    
    # Add grid
    ax.grid(True, alpha=0.3)

def draw_enhanced_cone_and_setup(ax, setup: Setup, n_generators=16):
    """Draw projection cone with enhanced visualization."""
    
    # Draw torch as a bright star
    ax.scatter([setup.T[0]], [setup.T[1]], [setup.T[2]], 
              marker='*', s=200, c=COLORS['torch'], 
              edgecolors='orange', linewidth=2, alpha=0.9)
    ax.text(setup.T[0], setup.T[1], setup.T[2] - 0.3, 
           'TORCH', ha='center', fontsize=10, weight='bold', color='orange')
    
    # Calculate wall center and basis vectors
    wall_center = setup.wall_n * setup.wall_d
    
    # Create orthonormal basis on wall
    u = unit(np.cross(setup.wall_n, np.array([1.0, 0.0, 0.0])))
    if np.linalg.norm(u) < 1e-6:
        u = unit(np.cross(setup.wall_n, np.array([0.0, 1.0, 0.0])))
    v = unit(np.cross(setup.wall_n, u))
    
    # Draw wall as a square
    wall_size = setup.base_radius * 1.5
    wall_corners = []
    for sx, sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
        corner = wall_center + wall_size * (sx*u + sy*v)
        wall_corners.append(corner)
    
    # Draw wall face
    wall_face = np.array(wall_corners)
    wall_poly = [[wall_face[0], wall_face[1], wall_face[2], wall_face[3]]]
    ax.add_collection3d(Poly3DCollection(wall_poly, alpha=0.3, 
                                        facecolor=COLORS['wall'], 
                                        edgecolor='navy', linewidth=1))
    
    # Draw wall normal vector
    normal_end = wall_center + setup.wall_n * 0.8
    ax.quiver(wall_center[0], wall_center[1], wall_center[2],
             setup.wall_n[0], setup.wall_n[1], setup.wall_n[2],
             color='navy', alpha=0.8, arrow_length_ratio=0.1)
    ax.text(normal_end[0], normal_end[1], normal_end[2], 
           'n', fontsize=10, color='navy', weight='bold')
    
    # Draw projection cone base circle
    theta = np.linspace(0, 2*np.pi, 50)
    circle_points = []
    for t in theta:
        point = wall_center + setup.base_radius * (np.cos(t)*u + np.sin(t)*v)
        circle_points.append(point)
    
    circle_points = np.array(circle_points)
    ax.plot(circle_points[:,0], circle_points[:,1], circle_points[:,2], 
           color=COLORS['cone'], linewidth=2, alpha=0.8)
    
    # Draw cone generators (lines from torch to circle)
    generator_indices = np.linspace(0, len(circle_points)-1, n_generators, dtype=int)
    for idx in generator_indices:
        point = circle_points[idx]
        ax.plot([setup.T[0], point[0]], 
               [setup.T[1], point[1]], 
               [setup.T[2], point[2]], 
               color=COLORS['cone'], linewidth=1, alpha=0.6)
    
    # Draw cone side surface (simplified)
    cone_surface = []
    for i in range(0, len(circle_points), 4):
        triangle = [setup.T, circle_points[i], circle_points[(i+1) % len(circle_points)]]
        cone_surface.append(triangle)
    
    ax.add_collection3d(Poly3DCollection(cone_surface, alpha=0.1, 
                                        facecolor=COLORS['cone'], 
                                        edgecolor='none'))

def plot_1_setup_overview():
    """Create comprehensive setup overview figure."""
    setup = Setup(
        T=np.array([0.0, 0.0, -3.0]),
        wall_n=np.array([0.0, 0.0, 1.0]),
        wall_d=5.0,
        base_radius=2.5,
    )
    
    L = generate_realistic_hand()
    # Position hand in the cone
    for i in L:
        L[i] = L[i] + np.array([0.0, 0.0, 1.0])  # Move forward into cone
    
    fig = plt.figure(figsize=(14, 10))
    
    # Main 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    draw_enhanced_cone_and_setup(ax1, setup)
    draw_enhanced_hand_3d(ax1, L, annotate_landmarks=False, title="Shadow Puppet Setup Overview")
    draw_coordinate_system(ax1, scale=1.0)
    
    # Side view (YZ plane)
    ax2 = fig.add_subplot(222)
    # Project to YZ plane
    y_coords = [L[i][1] for i in L]
    z_coords = [L[i][2] for i in L]
    ax2.scatter(y_coords, z_coords, s=30, alpha=0.7)
    
    # Show torch and wall in side view
    ax2.scatter([setup.T[1]], [setup.T[2]], marker='*', s=150, c=COLORS['torch'])
    ax2.axvline(x=setup.wall_d, color=COLORS['wall'], linewidth=3, alpha=0.7, label='Wall')
    
    # Draw cone outline in side view
    cone_top_y = setup.base_radius
    cone_bottom_y = -setup.base_radius
    ax2.plot([setup.T[1], cone_top_y], [setup.T[2], setup.wall_d], 
            color=COLORS['cone'], linewidth=2, alpha=0.7)
    ax2.plot([setup.T[1], cone_bottom_y], [setup.T[2], setup.wall_d], 
            color=COLORS['cone'], linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Y (cm)')
    ax2.set_ylabel('Z (cm)')
    ax2.set_title('Side View (YZ plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Top view (XY plane)
    ax3 = fig.add_subplot(223)
    x_coords = [L[i][0] for i in L]
    y_coords = [L[i][1] for i in L]
    ax3.scatter(x_coords, y_coords, s=30, alpha=0.7)
    
    # Draw hand connections in top view
    for (i,j) in ALL_CONNECTIONS:
        if i in L and j in L:
            ax3.plot([L[i][0], L[j][0]], [L[i][1], L[j][1]], 'k-', alpha=0.5)
    
    ax3.scatter([setup.T[0]], [setup.T[1]], marker='*', s=150, c=COLORS['torch'])
    
    # Draw cone base circle in top view
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = setup.base_radius * np.cos(theta)
    circle_y = setup.base_radius * np.sin(theta)
    ax3.plot(circle_x, circle_y, color=COLORS['cone'], linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('X (cm)')
    ax3.set_ylabel('Y (cm)')
    ax3.set_title('Top View (XY plane)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Parameter summary
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # Create parameter summary text
    param_text = f"""
SETUP PARAMETERS:
• Torch position: ({setup.T[0]:.1f}, {setup.T[1]:.1f}, {setup.T[2]:.1f}) cm
• Wall normal: ({setup.wall_n[0]:.1f}, {setup.wall_n[1]:.1f}, {setup.wall_n[2]:.1f})
• Wall distance: {setup.wall_d:.1f} cm from origin
• Cone base radius: {setup.base_radius:.1f} cm

INVARIANCES:
• Scale invariance within cone
• Translation invariance within cone  
• Rotation ±90° perpendicular to torch-wall axis

PARAMETRIZATION:
• Palm plane (3 points): Wrist, Index MCP, Pinky MCP
• Thumb angle: Angle between thumb and palm planes
• 4 inter-finger angles: T-I, I-M, M-R, R-P
• Joint angles: 3 per finger (except thumb: 2)
"""
    
    ax4.text(0.05, 0.95, param_text, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "01_setup_overview.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_2_mediapipe_landmarks():
    """Create detailed MediaPipe landmark visualization."""
    L = generate_realistic_hand()
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D annotated view
    ax1 = fig.add_subplot(131, projection='3d')
    draw_enhanced_hand_3d(ax1, L, annotate_landmarks=True, 
                         title="MediaPipe Hand Landmarks (21 points)")
    
    # Add legend for landmark categories
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=8, label='Wrist (0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['thumb'], 
                  markersize=8, label='Thumb (1-4)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['index'], 
                  markersize=8, label='Index (5-8)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['middle'], 
                  markersize=8, label='Middle (9-12)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ring'], 
                  markersize=8, label='Ring (13-16)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pinky'], 
                  markersize=8, label='Pinky (17-20)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Anatomical labels view
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Draw hand without landmark numbers but with anatomical labels
    draw_enhanced_hand_3d(ax2, L, annotate_landmarks=False, 
                         title="Anatomical Joint Names")
    
    # Add anatomical labels
    anatomical_labels = {
        0: 'WRIST',
        1: 'T-CMC', 2: 'T-MCP', 3: 'T-IP', 4: 'T-TIP',
        5: 'I-MCP', 6: 'I-PIP', 7: 'I-DIP', 8: 'I-TIP',
        9: 'M-MCP', 10: 'M-PIP', 11: 'M-DIP', 12: 'M-TIP',
        13: 'R-MCP', 14: 'R-PIP', 15: 'R-DIP', 16: 'R-TIP',
        17: 'P-MCP', 18: 'P-PIP', 19: 'P-DIP', 20: 'P-TIP'
    }
    
    for i, label in anatomical_labels.items():
        if i in L:
            ax2.text(L[i][0], L[i][1], L[i][2], label, 
                    fontsize=6, color='black', weight='bold')
    
    # Landmark table
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    # Create detailed landmark table
    table_data = []
    for i in range(21):
        if i in L:
            finger = ''
            joint_type = ''
            
            if i == 0:
                finger = 'WRIST'
                joint_type = 'Base'
            elif 1 <= i <= 4:
                finger = 'THUMB'
                joint_types = ['CMC', 'MCP', 'IP', 'TIP']
                joint_type = joint_types[i-1]
            elif 5 <= i <= 8:
                finger = 'INDEX'
                joint_types = ['MCP', 'PIP', 'DIP', 'TIP']
                joint_type = joint_types[i-5]
            elif 9 <= i <= 12:
                finger = 'MIDDLE'
                joint_types = ['MCP', 'PIP', 'DIP', 'TIP']
                joint_type = joint_types[i-9]
            elif 13 <= i <= 16:
                finger = 'RING'
                joint_types = ['MCP', 'PIP', 'DIP', 'TIP']
                joint_type = joint_types[i-13]
            elif 17 <= i <= 20:
                finger = 'PINKY'
                joint_types = ['MCP', 'PIP', 'DIP', 'TIP']
                joint_type = joint_types[i-17]
            
            table_data.append([i, finger, joint_type, 
                             f"({L[i][0]:.1f}, {L[i][1]:.1f}, {L[i][2]:.1f})"])
    
    # Display table
    table_text = "ID | FINGER | JOINT | POSITION (x,y,z)\n"
    table_text += "-" * 45 + "\n"
    for row in table_data:
        table_text += f"{row[0]:2d} | {row[1]:6s} | {row[2]:4s} | {row[3]}\n"
    
    ax3.text(0.05, 0.95, table_text, transform=ax3.transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "02_mediapipe_landmarks.png", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Create the first two improved figures
    print("Generating improved visualizations...")
    plot_1_setup_overview()
    plot_2_mediapipe_landmarks()
    print("Generated: 01_setup_overview.png, 02_mediapipe_landmarks.png")
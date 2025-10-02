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
    Generate an optimized hand pose for clear visualization of all angles and planes.
    """
    # Load default parameters optimized for visualization
    default_params = {
        "phi_thumb": 35.0,  # Increased for better thumb plane visibility
        "inter": {"T_I": 45.0, "I_M": 25.0, "M_R": 20.0, "R_P": 25.0},  # More spread for clearer angles
        "joints": {
            "T": [20.0, 20.0, 15.0],  # Slightly curved thumb
            "I": [15.0, 18.0, 15.0],  # Moderate curl for visibility
            "M": [12.0, 15.0, 12.0],  # Straighter middle finger
            "R": [18.0, 20.0, 18.0],  # More curl for distinction
            "P": [20.0, 22.0, 20.0]   # Most curved for clear differentiation
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
    
    # Optimized MCP positions for better plane visualization
    # Create a more pronounced arc for the knuckles
    base_spread = 1.0  # Increased spread
    mcp_height = 1.4   # Forward position
    
    # MCP positions with better angular separation
    L[5] = np.array([-0.6 * base_spread, mcp_height, 0.1])    # Index MCP
    L[9] = np.array([0.0, mcp_height + 0.1, 0.0])             # Middle MCP (highest)
    L[13] = np.array([0.5 * base_spread, mcp_height, 0.0])    # Ring MCP
    L[17] = np.array([0.9 * base_spread, mcp_height - 0.2, -0.1])  # Pinky MCP (lower)
    
    # Thumb CMC position - positioned for optimal thumb plane visibility
    L[1] = np.array([-1.1 * base_spread, 0.6, 0.3])  # CMC with Z elevation
    
    # Generate finger positions using more realistic biomechanical constraints
    finger_lengths = {
        'T': [0.9, 0.7, 0.5],   # Thumb segments
        'I': [1.3, 1.0, 0.7],   # Index segments  
        'M': [1.5, 1.1, 0.8],   # Middle segments
        'R': [1.4, 1.0, 0.7],   # Ring segments
        'P': [1.1, 0.9, 0.6]    # Pinky segments
    }
    
    # Build thumb with optimized positioning for plane visibility
    thumb_base_dir = unit(np.array([0.8, 0.6, 0.4]))  # More pronounced 3D direction
    
    # Thumb MCP
    L[2] = L[1] + finger_lengths['T'][0] * thumb_base_dir
    
    # Apply thumb joint angles with rotation
    def rotate_around_axis(vector, axis, angle_deg):
        """Rotate vector around axis by angle_deg degrees."""
        angle_rad = np.radians(angle_deg)
        axis = unit(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return (vector * cos_a + 
                np.cross(axis, vector) * sin_a + 
                axis * np.dot(axis, vector) * (1 - cos_a))
    
    # Thumb IP and TIP with joint rotations
    thumb_dir = thumb_base_dir
    rotation_axis = unit(np.cross(thumb_dir, np.array([0, 0, 1])))
    
    # Thumb IP
    thumb_dir = rotate_around_axis(thumb_dir, rotation_axis, -default_params['joints']['T'][1])
    L[3] = L[2] + finger_lengths['T'][1] * thumb_dir
    
    # Thumb TIP
    thumb_dir = rotate_around_axis(thumb_dir, rotation_axis, -default_params['joints']['T'][2])
    L[4] = L[3] + finger_lengths['T'][2] * thumb_dir
    
    # Build other fingers with proper inter-finger angles and joint bends
    finger_data = [
        ('I', [5, 6, 7, 8], np.array([0.0, 1.0, 0.15])),    # Index: slight upward tilt
        ('M', [9, 10, 11, 12], np.array([0.0, 1.0, 0.05])), # Middle: mostly forward
        ('R', [13, 14, 15, 16], np.array([0.0, 1.0, 0.0])), # Ring: straight forward
        ('P', [17, 18, 19, 20], np.array([0.0, 1.0, -0.1])) # Pinky: slight downward
    ]
    
    for finger_name, chain, base_direction in finger_data:
        mcp_id = chain[0]
        joint_angles = default_params['joints'][finger_name]
        lengths = finger_lengths[finger_name]
        
        # Start from MCP position
        current_pos = L[mcp_id]
        current_dir = unit(base_direction)
        
        # Apply progressive joint bends
        for i, (joint_id, length, bend_angle) in enumerate(zip(chain[1:], lengths, joint_angles)):
            # Create rotation axis perpendicular to finger direction and palm normal
            palm_normal = np.array([0, 0, 1])  # Approximate palm normal
            rotation_axis = unit(np.cross(current_dir, palm_normal))
            
            # Apply joint bend
            current_dir = rotate_around_axis(current_dir, rotation_axis, -bend_angle)
            
            # Calculate next joint position
            L[joint_id] = current_pos + length * current_dir
            current_pos = L[joint_id]
    
    # Apply small rotations to create more natural inter-finger angles
    inter_angles = default_params['inter']
    
    # Adjust finger directions based on inter-finger angles
    # This is a simplified approach - in reality, this would affect the entire finger chain
    palm_center = (L[5] + L[9] + L[13] + L[17]) / 4
    
    # Fine-tune positions for better visualization
    for i in L:
        # Add small random variations for more natural look
        if i > 0:  # Don't move wrist
            noise = np.random.normal(0, 0.02, 3)  # Small noise
            L[i] = L[i] + noise
    
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
                         title=None,
                         view_angle='default'):
    """Draw hand with enhanced visualization including colors and proper annotations."""
    
    # Draw connections with finger-specific colors
    if show_finger_colors:
        for finger, connections in FINGER_CONNECTIONS.items():
            color = COLORS[finger]
            for (i,j) in connections:
                if i in L and j in L:
                    P, Q = L[i], L[j]
                    ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                           color=color, linewidth=3, alpha=0.9)
        
        # Draw palm connections
        for (i,j) in PALM_CONNECTIONS:
            if i in L and j in L:
                P, Q = L[i], L[j]
                ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                       color=COLORS['palm'], linewidth=2.5, alpha=0.8)
    else:
        # Draw all connections in single color
        for (i,j) in ALL_CONNECTIONS:
            if i in L and j in L:
                P, Q = L[i], L[j]
                ax.plot([P[0],Q[0]], [P[1],Q[1]], [P[2],Q[2]], 
                       'k-', linewidth=2, alpha=0.8)
    
    # Draw landmarks as colored spheres with better visibility
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
                  s=60, c=color, alpha=0.95, edgecolors='black', linewidth=1)
        
        # Annotate landmarks with better positioning
        if annotate_landmarks:
            offset = 0.08 if i == 0 else 0.06
            ax.text(L[i][0], L[i][1], L[i][2] + offset, f'{i}', 
                   fontsize=8, color='black', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Styling
    ax.set_xlabel("X (cm)", fontsize=11)
    ax.set_ylabel("Y (cm)", fontsize=11) 
    ax.set_zlabel("Z (cm)", fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=13, weight='bold', pad=20)
    
    # Set optimal viewing angles based on what we want to show
    if view_angle == 'palm_plane':
        ax.view_init(elev=5, azim=-45)  # Good for palm plane visibility
    elif view_angle == 'thumb_angle':
        ax.view_init(elev=20, azim=-30)  # Good for thumb-palm angle
    elif view_angle == 'inter_finger':
        ax.view_init(elev=25, azim=-60)  # Good for inter-finger angles
    elif view_angle == 'side':
        ax.view_init(elev=0, azim=0)    # Side view
    elif view_angle == 'top':
        ax.view_init(elev=90, azim=0)   # Top-down view
    else:  # default
        ax.view_init(elev=20, azim=-50)  # General optimal view
    
    set_equal_3d_aspect(ax, padding=0.15)
    
    # Add enhanced grid
    ax.grid(True, alpha=0.4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges more subtle
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

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

def draw_plane_with_points(ax, points, color='blue', alpha=0.3, size=2.0, label=None):
    """Draw a plane defined by three points with the points highlighted."""
    p1, p2, p3 = points
    
    # Calculate plane normal and center
    v1 = p2 - p1
    v2 = p3 - p1
    normal = unit(np.cross(v1, v2))
    center = (p1 + p2 + p3) / 3
    
    # Create orthonormal basis in plane
    u = unit(v1)
    v = unit(np.cross(normal, u))
    
    # Create plane patch
    plane_points = []
    for sx, sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
        point = center + size * (sx*u + sy*v)
        plane_points.append(point)
    
    plane_points = np.array(plane_points)
    
    # Draw plane as polygon
    plane_poly = [[plane_points[0], plane_points[1], plane_points[2], plane_points[3]]]
    ax.add_collection3d(Poly3DCollection(plane_poly, alpha=alpha, 
                                        facecolor=color, edgecolor=color))
    
    # Draw defining points
    for i, point in enumerate(points):
        ax.scatter([point[0]], [point[1]], [point[2]], 
                  s=80, c='red', edgecolors='darkred', linewidth=2)
        ax.text(point[0], point[1], point[2] + 0.1, f'P{i+1}', 
               fontsize=9, weight='bold', color='darkred')
    
    # Draw plane normal
    normal_end = center + normal * 0.8
    ax.quiver(center[0], center[1], center[2],
             normal[0], normal[1], normal[2],
             color=color, alpha=0.8, arrow_length_ratio=0.15, linewidth=2)
    
    if label:
        ax.text(center[0], center[1], center[2], label, 
               fontsize=10, weight='bold', color=color)
    
    return normal, center

def draw_angle_between_vectors(ax, v1, v2, origin, radius=0.5, color='red', label=None):
    """Draw an angle arc between two vectors."""
    v1_norm = unit(v1)
    v2_norm = unit(v2)
    
    # Calculate angle
    angle = angle_between(v1, v2)
    
    # Create arc points
    # Find a vector perpendicular to both for rotation axis
    if np.allclose(np.cross(v1_norm, v2_norm), 0):
        # Vectors are parallel
        return angle
    
    # Create arc in the plane of the two vectors
    n_points = 20
    angles = np.linspace(0, np.radians(angle), n_points)
    
    # Rotation axis
    rot_axis = unit(np.cross(v1_norm, v2_norm))
    
    arc_points = []
    for a in angles:
        # Rodrigues rotation formula
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        rotated = (v1_norm * cos_a + 
                  np.cross(rot_axis, v1_norm) * sin_a + 
                  rot_axis * np.dot(rot_axis, v1_norm) * (1 - cos_a))
        arc_points.append(origin + radius * rotated)
    
    arc_points = np.array(arc_points)
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
           color=color, linewidth=2)
    
    # Add angle label
    if label:
        mid_point = origin + radius * unit((v1_norm + v2_norm) / 2)
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
               f'{angle:.1f}°', fontsize=9, color=color, weight='bold')
    
    return angle

def plot_3_palm_plane_definition():
    """Visualize how the palm plane is defined by three key points."""
    L = generate_realistic_hand()
    
    fig = plt.figure(figsize=(16, 8))
    
    # 3D view showing palm plane definition
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw hand with optimal view for palm plane
    draw_enhanced_hand_3d(ax1, L, annotate_landmarks=False, 
                         title="Palm Plane Definition", view_angle='palm_plane')
    
    # Define palm plane using wrist, index MCP, and pinky MCP
    palm_points = [L[0], L[5], L[17]]  # Wrist, Index MCP, Pinky MCP
    palm_normal, palm_center = draw_plane_with_points(
        ax1, palm_points, color='blue', alpha=0.4, size=2.0, label='PALM PLANE')
    
    # Highlight the defining triangle with thicker lines
    triangle_points = np.array(palm_points + [palm_points[0]])  # Close triangle
    ax1.plot(triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2], 
            'r-', linewidth=4, alpha=0.9)
    
    # Add clear annotations with better positioning
    offset = 0.15
    ax1.text(L[0][0], L[0][1], L[0][2] - offset, 'WRIST (0)', 
            fontsize=11, weight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax1.text(L[5][0] - offset, L[5][1], L[5][2] + offset, 'INDEX MCP (5)', 
            fontsize=11, weight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax1.text(L[17][0] + offset, L[17][1], L[17][2] + offset, 'PINKY MCP (17)', 
            fontsize=11, weight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    draw_coordinate_system(ax1, scale=0.8)
    
    # 2D schematic view
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Create schematic diagram
    schematic_text = """
PALM PLANE PARAMETRIZATION

The palm plane is uniquely defined by three non-collinear points:

1. WRIST (Landmark 0)
   • Base of the hand structure
   • Origin for palm measurements

2. INDEX MCP (Landmark 5) 
   • Metacarpophalangeal joint of index finger
   • Lateral boundary of palm

3. PINKY MCP (Landmark 17)
   • Metacarpophalangeal joint of pinky finger  
   • Medial boundary of palm

MATHEMATICAL DEFINITION:
Given three points P₁(wrist), P₂(index MCP), P₃(pinky MCP):

• Plane normal: n = (P₂-P₁) × (P₃-P₁) / ||(P₂-P₁) × (P₃-P₁)||
• Plane equation: n·(x - P₁) = 0
• Plane center: c = (P₁ + P₂ + P₃) / 3

INVARIANCE PROPERTIES:
✓ Robust to finger position changes
✓ Captures main palm orientation
✓ Independent of finger curl/spread
✓ Stable across different hand poses
"""
    
    ax2.text(0.05, 0.95, schematic_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.9))
    
    # Add palm plane equation
    eq_text = f"""
CURRENT PALM PLANE:
Normal vector: ({palm_normal[0]:.3f}, {palm_normal[1]:.3f}, {palm_normal[2]:.3f})
Center point: ({palm_center[0]:.1f}, {palm_center[1]:.1f}, {palm_center[2]:.1f}) cm
"""
    ax2.text(0.05, 0.15, eq_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "03_palm_plane_definition.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_4_thumb_palm_angle():
    """Visualize the angle between thumb plane and palm plane."""
    L = generate_realistic_hand()
    
    fig = plt.figure(figsize=(16, 8))
    
    # 3D view showing both planes and their angle
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw hand with optimal view for thumb angle
    draw_enhanced_hand_3d(ax1, L, annotate_landmarks=False, 
                         title="Thumb-Palm Angle (φ_thumb)", view_angle='thumb_angle')
    
    # Define palm plane
    palm_points = [L[0], L[5], L[17]]
    palm_normal, palm_center = draw_plane_with_points(
        ax1, palm_points, color='blue', alpha=0.25, size=1.8, label='PALM')
    
    # Define thumb plane using thumb landmarks and palm center
    palm_middle = (L[9] + L[13]) / 2  # Center between middle and ring MCPs
    thumb_points = [L[1], L[2], palm_middle]  # Thumb CMC, MCP, palm center
    thumb_normal, thumb_center = draw_plane_with_points(
        ax1, thumb_points, color='red', alpha=0.25, size=1.4, label='THUMB')
    
    # Calculate and display angle between planes
    phi_thumb = angle_between(palm_normal, thumb_normal)
    
    # Draw angle between normals with larger arc for visibility
    draw_angle_between_vectors(ax1, palm_normal, thumb_normal, 
                              palm_center, radius=0.8, color='green')
    
    # Add prominent angle label
    mid_normal = unit((palm_normal + thumb_normal) / 2)
    angle_label_pos = palm_center + 0.9 * mid_normal
    ax1.text(angle_label_pos[0], angle_label_pos[1], angle_label_pos[2], 
             f'φ_thumb = {phi_thumb:.1f}°', 
             fontsize=12, weight='bold', color='green',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9))
    
    # Highlight thumb chain with thicker lines
    thumb_chain = [1, 2, 3, 4]
    for i in range(len(thumb_chain)-1):
        p1, p2 = L[thumb_chain[i]], L[thumb_chain[i+1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='red', linewidth=5, alpha=0.9)
    
    draw_coordinate_system(ax1, scale=0.8)
    
    # Analysis view
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Create analysis diagram
    analysis_text = f"""
THUMB-PALM ANGLE ANALYSIS

CURRENT MEASUREMENT: φ_thumb = {phi_thumb:.1f}°

THUMB PLANE DEFINITION:
• Thumb CMC (Landmark 1): Carpometacarpal joint
• Thumb MCP (Landmark 2): Metacarpophalangeal joint  
• Palm center: Midpoint of middle-ring MCPs

PALM PLANE DEFINITION:
• Wrist (Landmark 0)
• Index MCP (Landmark 5)
• Pinky MCP (Landmark 17)

ANGLE CALCULATION:
φ_thumb = arccos(n_palm · n_thumb)

Where:
• n_palm = normalized palm plane normal
• n_thumb = normalized thumb plane normal

TYPICAL RANGES:
• Relaxed hand: 15-30°
• Thumb opposition: 45-80°
• Flat hand: 5-15°
• Gripping pose: 30-60°

SIGNIFICANCE:
This angle captures the fundamental thumb position
relative to the palm, crucial for shadow puppet
shape determination and hand pose classification.
"""
    
    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "04_thumb_palm_angle.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_5_interfinger_angles():
    """Visualize the four inter-finger angles with clear measurements."""
    L = generate_realistic_hand()
    
    fig = plt.figure(figsize=(18, 12))
    
    # Main 3D view with all angles
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Draw hand with optimal view for inter-finger angles
    draw_enhanced_hand_3d(ax1, L, annotate_landmarks=False, 
                         title="Inter-finger Angles", view_angle='inter_finger')
    
    # Define finger base directions with better visibility
    finger_directions = {
        'T': unit(L[2] - L[1]),      # Thumb: CMC to MCP
        'I': unit(L[6] - L[5]),      # Index: MCP to PIP
        'M': unit(L[10] - L[9]),     # Middle: MCP to PIP
        'R': unit(L[14] - L[13]),    # Ring: MCP to PIP
        'P': unit(L[18] - L[17])     # Pinky: MCP to PIP
    }
    
    finger_origins = {
        'T': L[1],   # Thumb CMC
        'I': L[5],   # Index MCP
        'M': L[9],   # Middle MCP
        'R': L[13],  # Ring MCP
        'P': L[17]   # Pinky MCP
    }
    
    # Draw finger direction vectors with enhanced visibility
    colors = {'T': COLORS['thumb'], 'I': COLORS['index'], 'M': COLORS['middle'], 
              'R': COLORS['ring'], 'P': COLORS['pinky']}
    
    for finger in finger_directions:
        origin = finger_origins[finger]
        direction = finger_directions[finger]
        end_point = origin + 1.5 * direction  # Longer vectors for better visibility
        
        ax1.quiver(origin[0], origin[1], origin[2],
                  1.5 * direction[0], 1.5 * direction[1], 1.5 * direction[2],
                  color=colors[finger], alpha=0.9, arrow_length_ratio=0.08, 
                  linewidth=4)
        ax1.text(end_point[0], end_point[1], end_point[2], 
                finger, fontsize=14, weight='bold', color=colors[finger],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Calculate and display inter-finger angles with larger arcs
    angle_pairs = [('T', 'I'), ('I', 'M'), ('M', 'R'), ('R', 'P')]
    angles = {}
    
    for i, (f1, f2) in enumerate(angle_pairs):
        origin = (finger_origins[f1] + finger_origins[f2]) / 2
        angle = draw_angle_between_vectors(ax1, finger_directions[f1], finger_directions[f2],
                                         origin, radius=0.6, color='red')
        angles[f'{f1}-{f2}'] = angle
        
        # Add angle text with better positioning
        mid_dir = unit((finger_directions[f1] + finger_directions[f2]) / 2)
        text_pos = origin + 0.7 * mid_dir
        ax1.text(text_pos[0], text_pos[1], text_pos[2], 
                f'{angle:.1f}°', fontsize=11, weight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    draw_coordinate_system(ax1, scale=0.6)
    
    # Individual angle views with better positioning
    views = [(222, 'T-I'), (223, 'I-M'), (224, 'M-R')]
    
    for subplot_idx, (ax_num, pair) in enumerate(views):
        ax = fig.add_subplot(ax_num, projection='3d')
        f1, f2 = pair.split('-')
        
        # Draw simplified view focusing on this pair
        draw_enhanced_hand_3d(ax, L, annotate_landmarks=False, 
                             show_finger_colors=False, title=f"Angle {pair}",
                             view_angle='inter_finger')
        
        # Highlight the two fingers with thicker vectors
        for finger in [f1, f2]:
            origin = finger_origins[finger]
            direction = finger_directions[finger]
            end_point = origin + 1.2 * direction
            
            ax.quiver(origin[0], origin[1], origin[2],
                     1.2 * direction[0], 1.2 * direction[1], 1.2 * direction[2],
                     color=colors[finger], alpha=0.95, arrow_length_ratio=0.12,
                     linewidth=5)
            ax.text(end_point[0], end_point[1], end_point[2], 
                   finger, fontsize=12, weight='bold', color=colors[finger],
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Draw the angle with enhanced visibility
        origin = (finger_origins[f1] + finger_origins[f2]) / 2
        angle = angles[f'{f1}-{f2}']
        draw_angle_between_vectors(ax, finger_directions[f1], finger_directions[f2],
                                 origin, radius=0.7, color='red')
        
        # Add prominent angle text
        ax.text2D(0.05, 0.95, f'{angle:.1f}°', transform=ax.transAxes,
                 fontsize=16, weight='bold', color='red',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95))
        
        set_equal_3d_aspect(ax, padding=0.2)
    
    # Add R-P angle information in text with better formatting
    rp_angle = angles['R-P']
    fig.text(0.75, 0.28, f'R-P Angle:\n{rp_angle:.1f}°', 
             fontsize=14, weight='bold', color='red', ha='center',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcoral", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(FIGDIR / "05_interfinger_angles.png", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Create all improved figures
    print("Generating improved visualizations...")
    plot_1_setup_overview()
    print("✓ 01_setup_overview.png")
    
    plot_2_mediapipe_landmarks()
    print("✓ 02_mediapipe_landmarks.png")
    
    plot_3_palm_plane_definition()
    print("✓ 03_palm_plane_definition.png")
    
    plot_4_thumb_palm_angle()
    print("✓ 04_thumb_palm_angle.png")
    
    plot_5_interfinger_angles()
    print("✓ 05_interfinger_angles.png")
    
    print("\nGenerated 5 enhanced visualization figures in 'improved_figures' directory.")
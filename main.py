import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button

from geometry import GeometryHandler
from config import get_parser, get_engine

#nazwa on the nose ale szkoda zmieniac
def animate_realistic_aerodynamics(csv_file=None, X_com=0.7, engine_choice="barrowman", parser_choice="csv"):
    ######GEOM SETUP#######
    if csv_file and os.path.exists(csv_file):
        print(f"Loading custom geometry from: {csv_file}")

        current_parser = get_parser(parser_choice)

        geom = GeometryHandler.from_file(csv_file, parser=current_parser, x_com=X_com)

    else:
        print("Using default internal geometry.")
        x_points = [0.0, 0.05, 0.1, 1.1, 1.4]
        r_points = [0.0, 0.0866, 0.1, 0.1, 0.15]
        geom = GeometryHandler(x_points, r_points, x_com=X_com)

    physics = get_engine(engine_choice, geometry=geom)

    print(f"Using Engine: {engine_choice.upper()}")
    print(f"Center of Mass (COM): {geom.x_com:.3f} m from nose tip")
    X_com = geom.x_com

    Q = 0.5 * 1.225 * 30.0 ** 2  # Dynamic Pressure
    Cd = 1.2  # Drag coef

    alphas_deg_sweep = np.linspace(-90, 90, 300)
    alphas_rad_sweep = np.radians(alphas_deg_sweep)

    CP_sweep = []
    F_normal_sweep = []
    Torque_sweep = []

    for a_rad in alphas_rad_sweep:
        cp, f_n, torque = physics.get_instant_state(a_rad, Q, Cd)
        CP_sweep.append(cp)
        F_normal_sweep.append(f_n)
        Torque_sweep.append(torque)

    CP_sweep = np.array(CP_sweep)
    F_normal_sweep = np.array(F_normal_sweep)
    Torque_sweep = np.array(Torque_sweep)

    max_F = np.max(np.abs(F_normal_sweep))

    ###### PLOT SETUP #######
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    fig.canvas.manager.set_window_title('Realistic Aerodynamic Stability')

    ax_anim = fig.add_subplot(gs[:, 0])
    max_dim = max(np.max(np.abs(geom.shape_x)), np.max(np.abs(geom.shape_y))) * 1.5
    ax_anim.set_xlim(-max_dim, max_dim)
    ax_anim.set_ylim(-max_dim, max_dim)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, linestyle='--', alpha=0.5)
    ax_anim.set_title("Physical Simulation (Falling)", fontsize=14)

    poly = Polygon(np.column_stack([geom.shape_x, geom.shape_y]), closed=True, fill=True, color='lightgray', ec='black')
    ax_anim.add_patch(poly)

    com_marker, = ax_anim.plot([], [], 'ko', markersize=8, label='COM (Fixed)')
    cp_marker, = ax_anim.plot([], [], 'mo', markersize=8, label='CP')
    force_arrow = ax_anim.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='red', lw=2.5))

    info_text = ax_anim.text(-max_dim * 0.9, max_dim * 0.8, '', fontsize=11, family='monospace',
                             bbox=dict(facecolor='white', alpha=0.9))
    ax_anim.legend(loc='lower left')

    ax_cp = fig.add_subplot(gs[0, 1])
    ax_cp.plot(alphas_deg_sweep, CP_sweep, color='purple', lw=2)
    ax_cp.axhline(X_com, color='black', ls='--', label='Center of Mass (COM)')
    ax_cp.fill_between(alphas_deg_sweep, CP_sweep, X_com, where=(CP_sweep < X_com), color='red', alpha=0.1)
    ax_cp.fill_between(alphas_deg_sweep, CP_sweep, X_com, where=(CP_sweep > X_com), color='green', alpha=0.1)
    ax_cp.set_ylabel('Position from Nose Tip (m)')
    ax_cp.set_title('Center of Pressure vs Center of Mass')
    ax_cp.grid(True, ls='--', alpha=0.5)
    cp_tracker_dot, = ax_cp.plot([], [], 'ro', markersize=10, markeredgecolor='black')

    ax_torque = fig.add_subplot(gs[1, 1], sharex=ax_cp)
    ax_torque.plot(alphas_deg_sweep, Torque_sweep, color='blue', lw=2)
    ax_torque.axhline(0, color='black', lw=1)
    ax_torque.fill_between(alphas_deg_sweep, 0, Torque_sweep, where=(Torque_sweep * alphas_deg_sweep < 0),
                           color='green', alpha=0.2, label='Stable')
    ax_torque.fill_between(alphas_deg_sweep, 0, Torque_sweep, where=(Torque_sweep * alphas_deg_sweep > 0), color='red',
                           alpha=0.2, label='Unstable')
    ax_torque.set_xlabel('Angle of Attack (Degrees)')
    ax_torque.set_ylabel('Torque (Nm)')
    ax_torque.grid(True, ls='--', alpha=0.5)
    ax_torque.legend(loc='upper right')
    torque_tracker_dot, = ax_torque.plot([], [], 'ro', markersize=10, markeredgecolor='black')

    plt.tight_layout()

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    ####### WIDGETS SETUP ###########
    #Slider Axi
    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    angle_slider = Slider(
        ax=ax_slider,
        label='Angle (°)',
        valmin=-60.0,
        valmax=60.0,
        valinit=0.0,
        valstep=0.1,
        color='blue'
    )

    ax_button = fig.add_axes([0.45, 0.02, 0.1, 0.04])
    play_button = Button(ax_button, 'Pause', hovercolor='0.975')



    ###### UPDATE FUNCTION #########
    physics_cache = {}

    def update(val):
        alpha_deg = round(angle_slider.val, 1)
        alpha_rad = np.radians(alpha_deg)

        if alpha_deg in physics_cache:
            CP_actual, F_normal, Torque = physics_cache[alpha_deg]
        else:
            CP_actual, F_normal, Torque = physics.get_instant_state(alpha_rad, Q, Cd)
            physics_cache[alpha_deg] = (CP_actual, F_normal, Torque)

        cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        rotated_coords = np.dot(np.column_stack([geom.shape_x, geom.shape_y]), rot_matrix)
        poly.set_xy(rotated_coords)

        cp_vector = np.array([0.0, CP_actual - X_com])
        cp_rotated = np.dot(cp_vector, rot_matrix)

        com_marker.set_data([0], [0])
        cp_marker.set_data([cp_rotated[0]], [cp_rotated[1]])

        if max_F > 0:
            force_scale = (max_dim * 0.5) / max_F
        else:
            force_scale = 0

        force_dx = F_normal * force_scale * np.cos(alpha_rad)
        force_dy = F_normal * force_scale * np.sin(alpha_rad)

        force_arrow.xy = (cp_rotated[0], cp_rotated[1])
        force_arrow.set_position((cp_rotated[0] - force_dx, cp_rotated[1] - force_dy))

        #
        is_stable = (Torque * alpha_deg < 0) or (abs(alpha_deg) < 0.1)
        status = "STABLE" if is_stable else "UNSTABLE"
        info_text.set_text(f"Angle:  {alpha_deg:>6.1f}°\nTorque: {Torque:>6.3f} Nm\nStatus: {status}")
        info_text.set_color("green" if is_stable else "red")

        cp_tracker_dot.set_data([alpha_deg], [CP_actual])
        torque_tracker_dot.set_data([alpha_deg], [Torque])

        fig.canvas.draw_idle()

    angle_slider.on_changed(update)

    animation_running = True
    anim_direction = 1.0

    def toggle_play(event):
        nonlocal animation_running
        animation_running = not animation_running
        if animation_running:
            play_button.label.set_text('Pause')
            ani.resume()
        else:
            play_button.label.set_text('Play')
            ani.pause()
        fig.canvas.draw_idle()

    play_button.on_clicked(toggle_play)

    def animation_step(frame):
        nonlocal anim_direction

        current_angle = angle_slider.val
        new_angle = current_angle + anim_direction

        if new_angle >= 60.0:
            new_angle = 60.0
            anim_direction = -1.0
        elif new_angle <= -60.0:
            new_angle = -60.0
            anim_direction = 1.0

        angle_slider.set_val(new_angle)

    # Creates the animation object
    ani = animation.FuncAnimation(
        fig,
        animation_step,
        interval=4.166,
        cache_frame_data=False
    )

    update(0)
    plt.show()


if __name__ == "__main__":
    FILE = "dart_shape.csv"
    CENTER_OF_MASS = "auto"
    """dla rocket_tube 0.7 dla innych auto"""

    #set up line
    animate_realistic_aerodynamics(
        csv_file=FILE,
        X_com=CENTER_OF_MASS,
        engine_choice="barrowman",
        parser_choice="csv"
    )
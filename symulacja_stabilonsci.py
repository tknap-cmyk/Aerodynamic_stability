import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button

#geom + phy
class AxisymmetricBody:
    def __init__(self, x_points, r_points, x_com="auto"):
        self.x = np.array(x_points, dtype=float)
        self.r = np.array(r_points, dtype=float)

        #Auto-COM calculator #
        if x_com == "auto" or x_com is None:
            self.x_com = self._calculate_volumetric_com()
        else:
            self.x_com = float(x_com)

        self.R_ref = np.max(self.r)
        self.A_ref = np.pi * (self.R_ref ** 2)

        self._calculate_barrowman()
        self._calculate_crossflow()
        self._build_2d_shape()

    @classmethod
    def from_csv(cls, filepath, x_com="auto"):
        import csv

        # READ header skipper itp.
        x_raw, r_raw = [], []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x_raw.append(float(row[0]))
                        r_raw.append(float(row[1]))
                    except ValueError:
                        continue

        x_raw = np.array(x_raw)
        r_raw = np.array(r_raw)

        #Airfoil Format Detection (Airfoils usually start at X=100 and go to X=0), now it is a bit obsolete in this version of code
        if x_raw[0] > x_raw[len(x_raw) // 2]:
            nose_idx = np.argmin(x_raw)

            # Extract just the top surface (from nose to tail)
            x_top = x_raw[:nose_idx + 1][::-1]  # Reverse for 0 -> Max
            r_top = r_raw[:nose_idx + 1][::-1]

            x_points = x_top
            r_points = r_top
        else:
            x_points = x_raw
            r_points = r_raw

        x_points = x_points - x_points[0]
        r_points = np.abs(r_points)

        return cls(x_points=x_points, r_points=r_points, x_com=x_com)

    def _calculate_volumetric_com(self):
        """COM assuming uniform density (Volumetric Centroid)."""
        total_volume = 0.0
        weighted_x = 0.0

        for i in range(len(self.x) - 1):
            x1, r1 = self.x[i], self.r[i]
            x2, r2 = self.x[i + 1], self.r[i + 1]
            h = x2 - x1
            if h <= 0: continue

            v_i = (np.pi * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

            if r1 == 0 and r2 == 0:
                local_centroid = h / 2.0
            else:
                local_centroid = (h / 4.0) * (r1 ** 2 + 2 * r1 * r2 + 3 * r2 ** 2) / (r1 ** 2 + r1 * r2 + r2 ** 2)

            x_com_i = x1 + local_centroid

            total_volume += v_i
            weighted_x += v_i * x_com_i

        # Avoid division by zero if geometry is flat
        return weighted_x / total_volume if total_volume > 0 else self.x[-1] / 2.0

    def _calculate_barrowman(self):
        total_CN = 0.0
        weighted_CP = 0.0

        max_r_idx = np.argmax(self.r)

        for i in range(len(self.x) - 1):
            x1, r1 = self.x[i], self.r[i]
            x2, r2 = self.x[i + 1], self.r[i + 1]
            L = x2 - x1
            if L <= 0: continue

            #Standard Barrowman Normal Force for przedzial
            CN_i = 2 * ((r2 / self.R_ref) ** 2 - (r1 / self.R_ref) ** 2)

            if i >= max_r_idx and r2 < r1:
                CN_i = 0.0

                #local CP for the przedzial
            if r1 + r2 == 0:
                X_cp_i = x1 + (L / 2)
            else:
                X_cp_i = x1 + (L / 3) * (r1 + 2 * r2) / (r1 + r2)

            total_CN += CN_i
            weighted_CP += CN_i * X_cp_i

        self.CN_low = total_CN #tu bylo powturzenie ale dziala
        self.CP_low = weighted_CP / total_CN if total_CN != 0 else self.x[-1] / 2

    def _calculate_crossflow(self):
        total_Area = 0.0
        weighted_Area_X = 0.0

        for i in range(len(self.x) - 1):
            x1, r1 = self.x[i], self.r[i]
            x2, r2 = self.x[i + 1], self.r[i + 1]
            L = x2 - x1

            A_i = 2 * ((r1 + r2) / 2) * L

            if r1 + r2 == 0:
                X_area_i = x1 + (L / 2)
            else:
                X_area_i = x1 + (L / 3) * (r1 + 2 * r2) / (r1 + r2)

            total_Area += A_i
            weighted_Area_X += A_i * X_area_i

        self.A_total = total_Area
        self.CP_high = weighted_Area_X / total_Area if total_Area != 0 else self.x[-1] / 2

    def _build_2d_shape(self):
        right_x = self.r
        right_y = self.x - self.x_com

        left_x = -self.r[::-1]
        left_y = self.x[::-1] - self.x_com

        self.shape_x = np.concatenate((right_x, left_x))
        self.shape_y = np.concatenate((right_y, left_y))

    def get_instant_state(self, alpha_rad, Q_dyn, Cd):

        #1 Barrowman Aerodynamic Lift (low angles of attack)
        F_lift = Q_dyn * self.A_ref * self.CN_low * np.sin(alpha_rad)

        #2 Newtonian Crossflow Drag(high angles of attack)
        F_cross = Q_dyn * self.A_total * Cd * (np.sin(alpha_rad) * np.abs(np.sin(alpha_rad)))

        F_normal = F_lift + F_cross

        #3 Calculate Restoring Torque
        Torque_lift = -F_lift * (self.CP_low - self.x_com)
        Torque_cross = -F_cross * (self.CP_high - self.x_com)

        Torque = Torque_lift + Torque_cross

        """4. Calculate Effective Center of Pressure (For Visualization)
         We work backward from the Torque to find where a single equivalent
         force would have to push to create that same Torque."""
        if F_normal != 0:
            CP_actual = self.x_com - (Torque / F_normal)
        else:
            #at 0, the CP is theoretically the Barrowman CP.
            CP_actual = self.CP_low

        return CP_actual, F_normal, Torque


def animate_realistic_aerodynamics(csv_file=None, X_com=0.7):
    ######GEOM SETUP#######
    if csv_file and os.path.exists(csv_file):
        print(f"Loading custom geometry from: {csv_file}")
        body = AxisymmetricBody.from_csv(csv_file, x_com=X_com)
    else:
        print("Using default internal geometry.")
        x_points = [0.0, 0.05, 0.1, 1.1, 1.4]
        r_points = [0.0, 0.0866, 0.1, 0.1, 0.15]
        body = AxisymmetricBody(x_points, r_points, x_com=X_com)

    print(f"AERODYNAMICS und MASS")
    print(f"Center of Mass (COM): {body.x_com:.3f} m from nose tip")
    print(f"Calculated CP_low:    {body.CP_low:.3f} m from nose tip")
    print(f"Calculated CP_high:   {body.CP_high:.3f} m from nose tip")


    X_com = body.x_com

    ###### PRECALCULATE FULL SWEEP #######
    Q = 0.5 * 1.225 * 30.0 ** 2  # Dynamic Pressure
    Cd = 1.2  # Drag coef

    alphas_deg_sweep = np.linspace(-90, 90, 300)
    alphas_rad_sweep = np.radians(alphas_deg_sweep)

    CP_sweep = []
    F_normal_sweep = []
    Torque_sweep = []

    #state for angle
    for a_rad in alphas_rad_sweep:
        cp, f_n, torque = body.get_instant_state(a_rad, Q, Cd)
        CP_sweep.append(cp)
        F_normal_sweep.append(f_n)
        Torque_sweep.append(torque)

    #lists to NumPy for Matplotlib
    CP_sweep = np.array(CP_sweep)
    F_normal_sweep = np.array(F_normal_sweep)
    Torque_sweep = np.array(Torque_sweep)

    max_F = np.max(np.abs(F_normal_sweep))

    ###### PLOT SETUP #######
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])
    fig.canvas.manager.set_window_title('Realistic Aerodynamic Stability')

    ax_anim = fig.add_subplot(gs[:, 0])
    #scaling anim window sorta dziala
    max_dim = max(np.max(np.abs(body.shape_x)), np.max(np.abs(body.shape_y))) * 1.5
    ax_anim.set_xlim(-max_dim, max_dim)
    ax_anim.set_ylim(-max_dim, max_dim)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True, linestyle='--', alpha=0.5)
    ax_anim.set_title("Physical Simulation (Falling)", fontsize=14)

    poly = Polygon(np.column_stack([body.shape_x, body.shape_y]), closed=True, fill=True, color='lightgray', ec='black')
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

    #room for slider and button
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

    # Play/Pause Button
    ax_button = fig.add_axes([0.45, 0.02, 0.1, 0.04])
    play_button = Button(ax_button, 'Pause', hovercolor='0.975')



    ###### UPDATE FUNCTION #########
    physics_cache = {}
    """tutaj ten cache jest i nie psuje ale tez do konca nie dziala."""
    def update(val):
        alpha_deg = round(angle_slider.val, 1) #prevents folating point
        alpha_rad = np.radians(alpha_deg)

        # CHECK CACHE po duplikaty aby nie liczyc za kazdym razem ale i tak program dziala wolno wiec cos nie tak
        if alpha_deg in physics_cache:
            CP_actual, F_normal, Torque = physics_cache[alpha_deg]
        else:
            CP_actual, F_normal, Torque = body.get_instant_state(alpha_rad, Q, Cd)

            physics_cache[alpha_deg] = (CP_actual, F_normal, Torque)

        # Rotate polygon
        cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        rotated_coords = np.dot(np.column_stack([body.shape_x, body.shape_y]), rot_matrix)
        poly.set_xy(rotated_coords)

        #Rotate and place markers
        cp_vector = np.array([0.0, CP_actual - X_com])
        cp_rotated = np.dot(cp_vector, rot_matrix)

        com_marker.set_data([0], [0])
        cp_marker.set_data([cp_rotated[0]], [cp_rotated[1]])

        # Force Arrow
        if max_F > 0:
            force_scale = (max_dim * 0.5) / max_F
        else:
            force_scale = 0

        force_dx = F_normal * force_scale * np.cos(alpha_rad)
        force_dy = F_normal * force_scale * np.sin(alpha_rad)

        force_arrow.xy = (cp_rotated[0], cp_rotated[1])
        force_arrow.set_position((cp_rotated[0] - force_dx, cp_rotated[1] - force_dy))

        # Text and Trackers
        is_stable = (Torque * alpha_deg < 0) or (abs(alpha_deg) < 0.1)
        status = "STABLE" if is_stable else "UNSTABLE"
        info_text.set_text(f"Angle:  {alpha_deg:>6.1f}°\nTorque: {Torque:>6.3f} Nm\nStatus: {status}")
        info_text.set_color("green" if is_stable else "red")

        cp_tracker_dot.set_data([alpha_deg], [CP_actual])
        torque_tracker_dot.set_data([alpha_deg], [Torque])

        fig.canvas.draw_idle()

    # Link slider to update function
    angle_slider.on_changed(update)

    ###### ANIMATION & PLAY/PAUSE LOGIC ########
    animation_running = True
    anim_direction = 1.0  # Controls whether we are adding or subtracting degrees

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
        nonlocal anim_direction  #Zmienia kierunek animacji

        current_angle = angle_slider.val
        new_angle = current_angle + anim_direction

        if new_angle >= 60.0:
            new_angle = 60.0
            anim_direction = -1.0  # Reverse direction to backwards
        elif new_angle <= -60.0:
            new_angle = -60.0
            anim_direction = 1.0  # Reverse direction to forwards

        angle_slider.set_val(new_angle)

    # Create the animation object
    ani = animation.FuncAnimation(
        fig,
        animation_step,
        interval=4.166,  #frame timing
        cache_frame_data=False
    )
    # Run once to initialize graphics
    update(0)

    plt.show()


if __name__ == "__main__":

    FILE = "rocket_tube.csv"
    # Set to a number (e.g., 0.7) for manual override, or "auto" for uniform density
    CENTER_OF_MASS = 0.7#"auto"

    animate_realistic_aerodynamics(csv_file=FILE, X_com=CENTER_OF_MASS)
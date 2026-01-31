import numpy as np
import matplotlib.pyplot as plt
from Tests.interfaceTracking import InterfaceTracker
from Scripts.stefan_simulation import setUpMesh, formfunction


class GridTimeIndependenceStudy:
    def __init__(self, InterfaceTracker, setUpMesh, formfunction, Lx=0.1, Ly=0.1, shape="rectangular"):
        self.InterfaceTracker = InterfaceTracker
        self.setUpMesh = setUpMesh
        self.formfunction = formfunction
        self.Lx = Lx
        self.Ly = Ly
        self.shape = shape

    def build_initial_conditions(self, nx, ny, cold_bottom_T=257.15, base_T=273.15, init_ice_thickness=0.01):
        # mesh first
        X, Y = self.setUpMesh(nx, ny, self.Lx, self.formfunction, self.shape)

        # initial temp
        initial_temp = np.ones((ny, nx)) * base_T

        
        jmid = nx // 2
        ycol = Y[:, jmid]
        y_bottom = ycol.min()

        number_of_frozen_cells = np.sum(ycol <= y_bottom + init_ice_thickness)
        number_of_frozen_cells = max(1, min(int(number_of_frozen_cells), ny - 1))

        
        initial_temp[number_of_frozen_cells:, :] += 0.1
        initial_temp[:number_of_frozen_cells, :] -= 1.0
        initial_temp[0, :] = cold_bottom_T  # bottom boundary

        fl_field_init = np.ones((ny, nx))
        fl_field_init[:number_of_frozen_cells, :] = 0.0

        return X, Y, initial_temp, fl_field_init

    def run_case(self, nx, ny, dt, total_time, q, cold_bottom_T=257.15, base_T=273.15, init_ice_thickness=0.01):
        steps_no = int(np.round(total_time / dt))
        total_time = steps_no * dt  # enforce exact same end time on the dt grid


        X, Y, initial_temp, fl_init = self.build_initial_conditions(
            nx, ny,
            cold_bottom_T=cold_bottom_T,
            base_T=base_T,
            init_ice_thickness=init_ice_thickness
        )

        sim = self.InterfaceTracker(X, Y, initial_temp, dt, steps_no, q, fl_init)
        sim.run()

        t = np.array(sim.times)
        z = np.array(sim.interface_positions)
        return t, z

    def plot_grid_independence(self, grids, dt_fixed, total_time, q, **ic_kwargs):
        plt.figure(figsize=(9, 6))

        for (nx, ny) in grids:
            t, z = self.run_case(nx, ny, dt_fixed, total_time, q, **ic_kwargs)
            plt.plot(t, z, marker='o', markersize=3, label=f"{nx}x{ny}, dt={dt_fixed}s")

        plt.xlabel("Time (s)")
        plt.ylabel("Ice thickness (m)")
        plt.title("Grid Independence: Interface vs Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_timestep_convergence(self, dts, nx_fixed, ny_fixed, total_time, q, **ic_kwargs):
        # Run all cases
        results = {}
        for dt in dts:
            t, z = self.run_case(nx_fixed, ny_fixed, dt, total_time, q, **ic_kwargs)
            results[dt] = (t, z)

        # Choose smallest dt as reference
        dt_ref = min(dts)
        t_ref, z_ref = results[dt_ref]

        # Plot interface vs time (optional)
        plt.figure(figsize=(9, 6))
        for dt in dts:
            t, z = results[dt]
            plt.plot(t, z, marker='o', markersize=3, label=f"dt={dt}s, {nx_fixed}x{ny_fixed}")
        plt.xlabel("Time (s)")
        plt.ylabel("Ice thickness (m)")
        plt.title("Time-step Independence: Interface vs Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Compute error norms vs reference
        errs = []
        for dt in sorted(dts):
            if dt == dt_ref:
                continue
            t, z = results[dt]
            # Interpolate onto reference time grid
            z_i = np.interp(t_ref, t, z)
            err = np.linalg.norm(z_i - z_ref) / np.sqrt(len(t_ref))  # RMS error
            errs.append((dt, err))

        # Plot error vs dt (this is the convergence plot)
        dts_plot = [e[0] for e in errs]
        err_plot = [e[1] for e in errs]

        plt.figure(figsize=(7, 5))
        plt.loglog(dts_plot, err_plot, marker='o')
        plt.xlabel(r'$\Delta t$ (s)')
        plt.ylabel('RMS error in interface thickness (m)')
        plt.title('Time-step Convergence (vs smallest $\Delta t$ reference)')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()

study = GridTimeIndependenceStudy(
    InterfaceTracker,
    setUpMesh,
    formfunction,
    Lx=0.1,
    Ly=0.1,
    shape="rectangular"
)

# Grid independence
grids = [(3,128), (3,256), (3,512), (3,1024)]
study.plot_grid_independence(
    grids=grids,
    dt_fixed=10.0,
    total_time=500.0,
    q=[0,0,0,0]
)

# Time-step independence
dts = [1.25, 0.625, 0.3125] 

study.plot_timestep_convergence(
    dts=dts,
    nx_fixed=3,
    ny_fixed=256,
    total_time=100,
    q=[0,0,0,0]
)

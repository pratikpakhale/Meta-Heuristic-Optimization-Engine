import numpy as np

def goldstein_price_2d(x):
    x1, x2 = x
    return (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) * \
           (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

def schwefel_2d(x):
    return - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def schaffer_2d(x):
    x1, x2 = x
    num = np.sin(x1**2 - x2**2)**2 - 0.5
    den = (1 + 0.001*(x1**2 + x2**2))**2
    return 0.5 + num / den

def rosenbrock_10d(x):
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere_10d(x):
    x = np.array(x)
    return np.sum(x**2)

def ackley_10d(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - \
           np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin_10d(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def martin_gaddy_2d(x):
    x1, x2 = x
    return (x1 - x2)**2 + ((x1 + x2 - 10) / 3)**2

def easom_2d(x):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def griewank_10d(x):
    x = np.array(x)
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

benchmark_functions = {
    "Goldstein-Price": {
        "function": goldstein_price_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -2,
        "UPPER_BOUND": 2,
        "GLOBAL_OPTIMUM": 3
    },
    "Schwefel": {
        "function": schwefel_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -500,
        "UPPER_BOUND": 500,
        "GLOBAL_OPTIMUM": -837.658
    },
    "Schaffer": {
        "function": schaffer_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -100,
        "UPPER_BOUND": 100,
        "GLOBAL_OPTIMUM": 0
    },
    "Rosenbrock": {
        "function": rosenbrock_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -1.2,
        "UPPER_BOUND": 1.2,
        "GLOBAL_OPTIMUM": 0
    },
    "Sphere": {
        "function": sphere_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Ackley": {
        "function": ackley_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -32.768,
        "UPPER_BOUND": 32.768,
        "GLOBAL_OPTIMUM": 0
    },
    "Rastrigin": {
        "function": rastrigin_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -5.12,
        "UPPER_BOUND": 5.12,
        "GLOBAL_OPTIMUM": 0
    },
    "Martin-Gaddy": {
        "function": martin_gaddy_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": 0,
        "UPPER_BOUND": 10,
        "GLOBAL_OPTIMUM": 0
    },
    "Easom": {
        "function": easom_2d,
        "DIMENSION": 2,
        "LOWER_BOUND": -100,
        "UPPER_BOUND": 100,
        "GLOBAL_OPTIMUM": -1
    },
    "Griewank": {
        "function": griewank_10d,
        "DIMENSION": 10,
        "LOWER_BOUND": -600,
        "UPPER_BOUND": 600,
        "GLOBAL_OPTIMUM": 0
    }
}


def multiple_disk_clutch_brake(x):
    """
    Multiple disk clutch brake optimization with improved constraint handling
    """
    # Problem constants
    rho = 7.85e-6    # Density
    mu = 0.5         # Friction coefficient
    s = 1.5          # Safety factor
    N = 250          # RPM
    Mh = 3           # Max torque
    prz_max = 1      # Max pressure
    Vsr_max = 10     # Max sliding velocity
    Tmax = 3         # Max thickness
    Tmin = 1.5       # Min thickness
    Lmax = 30        # Max length
    Δr = 20          # Min radius difference
    pi = np.pi

    # Extract design variables
    ri, ro, t, F, n = x
    n = round(n)  # Integer number of disks

    # Prevent division by zero
    if ri >= ro or (ro**2 - ri**2) <= 0:
        return float('inf')

    # Objective function (minimize mass)
    fx = pi * (ro**2 - ri**2) * t * (n + 1) * rho

    # Calculate constraints
    try:
        g1 = (ro - ri) - Δr
        g2 = Lmax - (n + 1) * (t + 0.5)
        g3 = prz_max - (F / (pi * (ro**2 - ri**2)))
        g4 = Vsr_max - (2 * pi * N * (ro**3 - ri**3) / (90 * (ro**2 - ri**2)))
        g5 = Tmax - t
        g6 = t - Tmin
        g7 = F - s * Mh / (mu * n * (ro + ri) / 2)

        # Improved constraint handling using static penalties
        total_violation = 0
        epsilon = 1e-10  # Small constant to avoid numerical issues

        # Calculate violations
        violations = [
            max(-g1, 0),  # Geometric constraint
            max(-g2, 0),  # Length constraint
            max(-g3, 0),  # Pressure constraint
            max(-g4, 0),  # Sliding velocity constraint
            max(-g5, 0),  # Maximum thickness constraint
            max(-g6, 0),  # Minimum thickness constraint
            max(-g7, 0)   # Torque constraint
        ]

        # Sum up violations with different weights
        weights = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0]  # Higher weights for critical constraints
        for v, w in zip(violations, weights):
            total_violation += w * v

        if total_violation > 0:
            # Use logarithmic penalty to avoid extreme values
            return fx * (1 + np.log1p(total_violation))
        return fx

    except (ZeroDivisionError, RuntimeWarning):
        return float('inf')


benchmark_functions = {
 
    "Multiple Disk Clutch Brake": {
        "function": multiple_disk_clutch_brake,
        "DIMENSION": 5,
        "LOWER_BOUND": [55, 75, 1.5, 1000, 2],
        "UPPER_BOUND": [110, 110, 3, 3000, 9],
        "GLOBAL_OPTIMUM": 0,
        "is_constrained": True,
        "can_plot_3d": False
    }
}



def robot_gripper(x, iteration=0, max_iter=1000):
    """
    Optimized version of the robot gripper fitness function using vectorized operations.
    """
    # Constants
    Y_min, Y_max, Y_C = 50, 100, 150
    Z_max = 100
    
    # Extract parameters
    a, b, c, e, f, l, delta = x
    
    # Vectorized y function
    def y(x, z_values):
        if not isinstance(z_values, np.ndarray):
            z_values = np.array([z_values])
        
        g = np.sqrt((l - z_values)**2 + e**2)
        beta = np.arccos(np.clip((b**2 + g**2 - a**2) / (2 * b * g), -1, 1))
        return 2 * (e + f + c * np.sin(beta + delta))
    
    # Calculate y values for the entire z range at once
    z_range = np.linspace(0, Z_max, num=50)  # Reduced from 100 to 50 points for better performance
    y_values = y(x, z_range)
    
    # Calculate objective function
    Fy_max = np.max(y_values)
    Fy_min = np.min(y_values)
    objective = Fy_max - Fy_min
    
    # Calculate constraints more efficiently
    g = np.sqrt((l - Z_max)**2 + e**2)
    y_z_max = y(x, Z_max)[0]  # Get scalar value
    y_zero = y(x, 0)[0]       # Get scalar value
    
    constraints = np.array([
        Y_min - y_z_max,                    # g1(x) >= 0
        y_z_max,                            # g2(x) >= 0
        y_zero - Y_max,                     # g3(x) >= 0
        Y_C - y_zero,                       # g4(x) >= 0
        (a + b)**2 - l**2 - e**2,          # g5(x) >= 0
        (l - Z_max)**2 + (a - e)**2 - b**2, # g6(x) >= 0
        l - Z_max                           # g7(x) >= 0
    ])
    
    # Calculate violation using vectorized operations
    violation = np.sum(np.maximum(0, -constraints))
    
    # Dynamic penalty factor
    penalty_factor = 1e3 * (1 + (iteration / max_iter))
    penalty = penalty_factor * violation
    
    # Return final fitness value
    return objective + penalty


benchmark_functions = {
    "Robot Gripper": {
        "function": robot_gripper,
        "DIMENSION": 7,
        "LOWER_BOUND": [10, 10, 100, 0, 0, 100, 1],
        "UPPER_BOUND": [150, 150, 200, 50, 50, 300, 3.14],
        "GLOBAL_OPTIMUM": 0,
        "is_constrained": True,
        "can_plot_3d": False
    }
}





def step_cone_pulley(x):
    rho = 7200         # Density in kg/m^3
    a = 3              # Distance parameter in meters
    mu = 0.35          # Friction coefficient
    s = 1.75e6         # Safety factor in N/m^2
    t = 8e-3           # Thickness in meters
    w = 1              # Weight of the material (can adjust as needed)
    N = 10             # Reference speed for scaling (adjust if necessary)
    penalty_factor = 1e3  # Penalty factor for constraint violations


    # Extract design variables
    d1, d2, d3, d4, N1, N2, N3, N4 = x

    # Objective function (mass minimization)
    objective = (rho * w) * (
        d1**2 * (1 + (N1 / N)**2) +
        d2**2 * (1 + (N2 / N)**2) +
        d3**2 * (1 + (N3 / N)**2) +
        d4**2 * (1 + (N4 / N)**2)
    )

    # Constraint evaluations
    constraints = []
    
    # Equality constraints (h1, h2, h3)
    constraints.append(d1 - d2)  # C1 - C2 = 0
    constraints.append(d1 - d3)  # C1 - C3 = 0
    constraints.append(d1 - d4)  # C1 - C4 = 0

    # Inequality constraints (Ri >= 2 and Pi >= threshold)
    for i, (di, Ni) in enumerate(zip([d1, d2, d3, d4], [N1, N2, N3, N4])):
        # Tension ratio Ri
        Ri = np.exp(mu * (np.pi - 2 * np.arcsin((Ni / N) - 1) * (di / (2 * a))))
        constraints.append(Ri - 2)

        # Power transmitted Pi
        Pi = s * w * (1 - np.exp(-mu * (np.pi - 2 * np.arcsin((Ni / N) - 1) * (di / (2 * a))))) * (np.pi * di * Ni / 60)
        constraints.append(Pi - (0.75 * 745.6998))

    # Calculate total constraint violation
    violation = np.sum(np.maximum(0, -np.array(constraints)))

    # Penalized fitness
    penalized_fitness = objective + penalty_factor * violation
    return penalized_fitness





benchmark_functions = {
    "Step Cone Pulley": {
        "function": step_cone_pulley,
        "DIMENSION": 8,  # d1, d2, d3, d4, N1, N2, N3, N4
        "LOWER_BOUND": [1, 1, 1, 1, 1, 1, 1, 1],  # Example lower bounds for each variable
        "UPPER_BOUND": [10, 10, 10, 10, 10, 10, 10, 10],  # Example upper bounds for each variable
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # Since it's an 8-dimensional problem
    }
}


# def hydrodynamic_thrust_bearing(x):
#     """Hydrodynamic thrust bearing design optimization with improved constraint handling."""
#     # Constants from the problem statement
#     gamma = 0.0307
#     C = 0.5
#     n = -3.55
#     C1 = 10.04
#     Ws = 101000
#     Pmax = 1000
#     Delta_Tmax = 50
#     hmin = 0.001
#     g = 386.4
#     N = 750
#     pi = np.pi
    
#     # Design variables
#     Q, R, R0, mu, p = x

#     # Intermediate calculations
#     W = (pi * Pmax * (R**2 - R0**2)) / (2 * np.log(R / R0))
#     h = ((2 * pi * N / 60)**2 * 2 * pi * mu) / (9336 * gamma * (2 * (10**p - 560) - C1)) * (R**4 - R0**4) / 4
#     P_0 = (6 * mu * Q) / (pi * h**3) * np.log(R / R0)
#     Ef = 9336 * Q * gamma * (2 * (10**p - 560) - C1)
#     Delta_T = 2 * (10**p - 560) - C1
    
#     # Objective function
#     objective = (Q * P_0) / 0.7 + Ef
    
#     # Constraints
#     constraints = []
#     constraints.append(W - Ws)                         # g1: W - Ws >= 0
#     constraints.append(Pmax - P_0)                     # g2: Pmax - P0 >= 0
#     constraints.append(Delta_Tmax - Delta_T)           # g3: DeltaTmax - DeltaT >= 0
#     constraints.append(h - hmin)                       # g4: h - hmin >= 0
#     constraints.append(R - R0)                         # g5: R - R0 >= 0
#     constraints.append(0.001 - (gamma / (g * P_0)) * (Q / (2 * pi * R * h)))  # g6
#     constraints.append(5000 - (W / (pi * (R**2 - R0**2))))  # g7
    
#     # Calculate constraint violation
#     total_violation = np.sum(np.maximum(0, -np.array(constraints)))
    
#     # Penalty factor for constraint violation
#     penalty_factor = 1e3 * (1 + (1 / 1))  # Example dynamic penalty (set max_iter as needed)
#     penalty = penalty_factor * total_violation
    
#     # Modified fitness
#     fitness = objective + penalty
#     return fitness

# benchmark_functions = {
#     "Hydrodynamic Thrust Bearing": {
#         "function": hydrodynamic_thrust_bearing,
#         "DIMENSION": 5,  # Q, R, R0, mu, p
#         "LOWER_BOUND": [1, 1, 1, 1e-6, 1],  # Example lower bounds for each variable
#         "UPPER_BOUND": [16, 16, 16, 16e-6, 10],  # Example upper bounds for each variable
#         "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
#         "is_constrained": True,
#         "can_plot_3d": False  # Since it's a 5-dimensional problem
#     }
# }



def hydrodynamic_thrust_bearing(x):
    # Constants
    gamma = 0.0307
    C = 0.5
    n = -3.55
    C1 = 10.04
    Ws = 101000
    P_max = 1000
    delta_T_max = 50
    h_min = 0.001
    g = 386.4
    N = 750
    penalty_factor = 1e3  # Penalty factor for constraint violations
    
    # Extract design variables
    R, R0, Q, Po, h = x

    # Dependent calculations
    W = (np.pi * Po * (R**2 - R0**2)) / (2 * np.log(R / R0))
    Ef = 93360 * gamma * delta_T_max
    delta_T = 2 * (10**Po - 560)
    P = np.log10(np.log10(8.122e6 * 0.000016 + 0.8)) - C1
    h_calc = ((2 * np.pi * N / 60) * (2 * np.pi * 0.000016) / Ef) * ((R**4 - R0**4) / 4)

    # Objective function
    objective = Q / 0.7 + Ef

    # Constraints
    constraints = []
    constraints.append(W - Ws)                                  # g1(x): W - Ws >= 0
    constraints.append(P_max - Po)                              # g2(x): P_max - Po >= 0
    constraints.append(delta_T_max - delta_T)                   # g3(x): delta_T_max - delta_T >= 0
    constraints.append(h - h_min)                               # g4(x): h - h_min >= 0
    constraints.append(R - R0)                                  # g5(x): R - R0 >= 0
    constraints.append(0.001 - (gamma / (g * Po)) * (Q / (2 * np.pi * R * h)))  # g6(x) >= 0
    constraints.append(5000 - W / (np.pi * (R**2 - R0**2)))     # g7(x) >= 0

    # Calculate total constraint violation
    violation = np.sum(np.maximum(0, -np.array(constraints)))

    # Penalized fitness
    penalized_fitness = objective + penalty_factor * violation
    return penalized_fitness

# Benchmark configuration
benchmark_functions = {
    "Hydrodynamic Thrust Bearing": {
        "function": hydrodynamic_thrust_bearing,
        "DIMENSION": 5,  # R, R0, Q, Po, h
        "LOWER_BOUND": [1, 1, 1, 1, 1],  # Example lower bounds for each variable
        "UPPER_BOUND": [16, 16, 16, 16, 16],  # Example upper bounds for each variable
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # Since it's a 5-dimensional problem
    }
}



def rolling_element_bearing(x):
    # Given constants
    D = 160    # mm
    d = 90     # mm
    Bw = 30    # mm
    phi0 = 2 * np.pi
    fc = 37.91  # Constant from the objective

    # Design variables from input array `x`
    Z, Db, Dm, Kdmin, Kdmax, e, ε, s = x

    # Objective function
    if Db <= 25.4:
        Cd = fc * Z**2/3 * Db**1.8
    else:
        Cd = 3.647 * fc * Z**2/3 * Db**1.4

    # Constraint functions
    constraints = []

    # g1(X)
    constraints.append(phi0 / (2 * np.arcsin(Db / Dm)) - Z + 1)

    # g2(X)
    constraints.append(2 * Db - Kdmin * (D - d))

    # g3(X)
    constraints.append(Kdmax * (D - d) - 2 * Db)

    # g4(X)
    constraints.append(Bw - Db)

    # g5(X)
    constraints.append(Dm - 0.5 * (D + d))

    # g6(X)
    constraints.append((0.5 + e) * (D + d) - Dm)

    # g7(X)
    constraints.append(0.5 * (D - Dm - Db) - ε * Db)

    # g8(X)
    constraints.append(s - 0.515)

    # g9(X)
    constraints.append(0.515 - s)

    # Constraint violation (sum of all negative constraint values)
    penalty_factor = 1e3  # Penalty factor for constraint violations
    violation = np.sum(np.maximum(0, -np.array(constraints)))

    # Penalized fitness
    penalized_fitness = -Cd + penalty_factor * violation
    return penalized_fitness

# Example function registration for use in an optimization benchmark
benchmark_functions = {
    "Rolling Element Bearing": {
        "function": rolling_element_bearing,
        "DIMENSION": 8,  # Number of design variables
        "LOWER_BOUND": [10, 10, 120, 0.5, 0.6, 0.1, 0.1, 0.515],  # Example lower bounds
        "UPPER_BOUND": [50, 50, 150, 0.7, 0.85, 0.4, 0.2, 0.515],  # Example upper bounds
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # For 8-dimensional problems
    }
}


def belleville_spring(x):
    # Given constants
    Pmax = 5400     # lb
    delta_max = 0.2 # in.
    S = 200e3       # psi
    E = 30e6        # psi
    mu = 0.3
    H = 2.0         # in.
    Dmax = 12.01    # in.

    # Extract design variables from input array `x`
    De, Di, t = x

    # Calculated parameters
    K = De / Di
    alpha = (6 / (np.pi * np.log(K))) * ((K - 1) / K)**2
    beta = (6 / (np.pi * np.log(K))) * ((K - 1) / (np.log(K) - 1))
    gamma = (6 / (np.pi * np.log(K))) * ((K - 1) / 2)

    # Objective function (volume of the spring)
    objective = 0.07075 * np.pi * (De**2 - Di**2) * t

    # Constraint evaluations
    constraints = []

    # g1(x)
    h = H - t  # Effective height of the spring
    constraints.append(S - (4 * E * delta_max / ((1 - mu**2) * alpha * De**2)) * (beta * (h - delta_max / 2) + gamma * t))

    # g2(x)
    delta = delta_max  # Maximum compression
    constraints.append((4 * E * delta / ((1 - mu**2) * alpha * De**2)) * ((h - delta / 2) * t + t**3) - Pmax)

    # g3(x)
    delta_i = h / t * 0.3  # Assume δ_i is proportional to h/t
    constraints.append(delta_i - delta_max)

    # g4(x)
    constraints.append(H - h)

    # g5(x)
    constraints.append(Dmax - De)

    # g6(x)
    constraints.append(De - Di)

    # g7(x)
    constraints.append(0.3 - (h / (De - Di)))

    # Calculate total constraint violation
    penalty_factor = 1e3  # Penalty factor for constraint violations
    violation = np.sum(np.maximum(0, -np.array(constraints)))

    # Penalized fitness
    penalized_fitness = objective + penalty_factor * violation
    return penalized_fitness

# Example function registration for use in an optimization benchmark
benchmark_functions = {
    "Belleville Spring": {
        "function": belleville_spring,
        "DIMENSION": 3,  # Number of design variables
        "LOWER_BOUND": [5, 1, 0.05],  # Example lower bounds (adjust as needed)
        "UPPER_BOUND": [12, 11, 0.5],  # Example upper bounds (adjust as needed)
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # 3D visualization possible for 3-dimensional problems
    }
}


# ---------------------------------------------------------------------------------------- #


benchmark_functions = {
     "Multiple Disk Clutch Brake": {
        "function": multiple_disk_clutch_brake,
        "DIMENSION": 5,
        "LOWER_BOUND": [55, 75, 1.5, 1000, 2],
        "UPPER_BOUND": [110, 110, 3, 3000, 9],
        "GLOBAL_OPTIMUM": 0,
        "is_constrained": True,
        "can_plot_3d": False
    },
"Robot Gripper": {
        "function": robot_gripper,
        "DIMENSION": 7,
        "LOWER_BOUND": [10, 10, 100, 0, 0, 100, 1],
        "UPPER_BOUND": [150, 150, 200, 50, 50, 300, 3.14],
        "GLOBAL_OPTIMUM": 0,
        "is_constrained": True,
        "can_plot_3d": False
    },
    "Step Cone Pulley": {
        "function": step_cone_pulley,
        "DIMENSION": 8,  # d1, d2, d3, d4, N1, N2, N3, N4
        "LOWER_BOUND": [1, 1, 1, 1, 1, 1, 1, 1],  # Example lower bounds for each variable
        "UPPER_BOUND": [10, 10, 10, 10, 10, 10, 10, 10],  # Example upper bounds for each variable
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # Since it's an 8-dimensional problem
    },
    "Hydrodynamic Thrust Bearing": {
        "function": hydrodynamic_thrust_bearing,
        "DIMENSION": 5,  # R, R0, Q, Po, h
        "LOWER_BOUND": [1, 1, 1, 1, 1],  # Example lower bounds for each variable
        "UPPER_BOUND": [16, 16, 16, 16, 16],  # Example upper bounds for each variable
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # Since it's a 5-dimensional problem
    },

 "Rolling Element Bearing": {
        "function": rolling_element_bearing,
        "DIMENSION": 8,  # Number of design variables
        "LOWER_BOUND": [10, 10, 120, 0.5, 0.6, 0.1, 0.1, 0.515],  # Example lower bounds
        "UPPER_BOUND": [50, 50, 150, 0.7, 0.85, 0.4, 0.2, 0.515],  # Example upper bounds
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # For 8-dimensional problems
    },
        "Belleville Spring": {
        "function": belleville_spring,
        "DIMENSION": 3,  # Number of design variables
        "LOWER_BOUND": [5, 1, 0.05],  # Example lower bounds (adjust as needed)
        "UPPER_BOUND": [12, 11, 0.5],  # Example upper bounds (adjust as needed)
        "GLOBAL_OPTIMUM": 0,  # Set this if known, otherwise keep as None
        "is_constrained": True,
        "can_plot_3d": False  # 3D visualization possible for 3-dimensional problems
    }


}
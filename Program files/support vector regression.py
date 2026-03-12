import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

plt.close("all")

plt.rcParams.update({'text.usetex': False})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif',
        'Computer Modern Roman'
    ]
})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.close('all')


def train_svr():
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_excel("../data/filtered_Coordinates.xlsx")

    X = df[["LF_X", "F_Y", "RF_Z"]].values
    Y = df[["opt_X", "opt_Y", "opt_Z"]].values

    # Keep frame indices so we know which rows are in the test set
    indices = np.arange(len(X))

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------
    # You can expand these lists if you want a real grid search
    C_list = [1e5]
    gamma_list = [1]

    cv_mse = np.zeros((len(C_list), len(gamma_list)))
    test_mse_grid = np.zeros_like(cv_mse)

    best_cv_mse = np.inf
    best_cv_idx = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------
    # Grid search (select by CV)
    # --------------------------------------------------------
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):

            base_model = SVR(
                kernel="rbf",
                C=C,
                gamma=gamma
            )

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", MultiOutputRegressor(base_model))
            ])

            # ---- 5-fold CV MSE ----
            cv_errors = []
            for tr, va in kf.split(X_train):
                model.fit(X_train[tr], Y_train[tr])
                pred = model.predict(X_train[va])
                cv_errors.append(mean_squared_error(Y_train[va], pred))

            cv_mse[i, j] = np.mean(cv_errors)

            # ---- Test MSE for this hyperparam (for info only) ----
            model.fit(X_train, Y_train)
            pred_test_tmp = model.predict(X_test)
            test_mse_grid[i, j] = mean_squared_error(Y_test, pred_test_tmp)

            # ---- Track best CV ----
            if cv_mse[i, j] < best_cv_mse:
                best_cv_mse = cv_mse[i, j]
                best_cv_idx = (i, j)

            print(f"C={C}, gamma={gamma:.1e} | "
                  f"CV MSE={cv_mse[i,j]:.2f}, Test MSE={test_mse_grid[i,j]:.2f}")

    # --------------------------------------------------------
    # Train final model with best CV hyperparameters
    # --------------------------------------------------------
    best_C = C_list[best_cv_idx[0]]
    best_gamma = gamma_list[best_cv_idx[1]]

    base_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(base_model))
    ])

    final_model.fit(X_train, Y_train)
    Y_test_pred = final_model.predict(X_test)
    final_test_mse = mean_squared_error(Y_test, Y_test_pred)

    print("\n================ FINAL MODEL (CV-SELECTED) ================")
    print(f"Best C (CV)        : {best_C}")
    print(f"Best Gamma (CV)    : {best_gamma}")
    print(f"Best CV MSE        : {best_cv_mse:.2f}")
    print(f"Test MSE           : {final_test_mse:.2f}")

    # --------------------------------------------------------
    # TEST-ONLY predictions & absolute error for plotting
    # --------------------------------------------------------
    # Use original frame indices for x-axis, sorted in time/frame order
    order = np.argsort(idx_test)
    frames_test = idx_test[order]

    Y_test_sorted = Y_test[order]
    Y_test_pred_sorted = Y_test_pred[order]

    # Absolute error in mm (always positive)
    errors_test = np.abs(Y_test_pred_sorted - Y_test_sorted)

    # --------------------------------------------------------
    # MANUAL TICK OPTIONS (EDIT THESE IF YOU WANT)
    # --------------------------------------------------------
    # For each subplot k = 0 (x), 1 (y), 2 (z):
    #   - set to a list/array of tick values, or leave as None to use automatic ticks.
    #
    # Example:
    coord_yticks_list = [
        np.arange(-2000, 1, 400),  # ticks for x (mm)
        np.arange(-700,  701, 280),  # ticks for y (mm)
        np.arange(1800, 3601, 360),   # ticks for z (mm)
    ]
    err_yticks_list = [
        np.arange(0, 151, 30),         # |x error| ticks
        np.arange(0, 81, 16),         # |y error| ticks
        np.arange(0, 61, 12),         # |z error| ticks
    ]
  

    # Optional: if you also want to force axis limits (e.g. to end at 1300):
    # coord_ylim_list = [
    #     (-2000, 1500),   # limits for x (mm)
    #     (-500,  1000),   # limits for y (mm)
    #     (2000, 4000),    # limits for z (mm)
    # ]
    # err_ylim_list = [
    #     (0, 80),         # limits for |x error|
    #     (0, 80),         # limits for |y error|
    #     (0, 80),         # limits for |z error|
    # ]
    coord_ylim_list = [None, None, None]
    err_ylim_list   = [None, None, None]

    # --------------------------------------------------------
    # x, y, z versus frame (TEST ONLY) with absolute error overlay
    # --------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 4), sharex=True)

    coord_labels = ["x (mm)", "y (mm)", "z (mm)"]
    error_labels = ["|error| (mm)", "|error| (mm)", "|error| (mm)"]

    for k, ax in enumerate(axes):
        # reference & predicted coordinates (only test data)
        line_ref, = ax.plot(
            frames_test, Y_test_sorted[:, k],
            linestyle='-',
            linewidth=0.8,
            label="reference OptiTrack coordinates",
            #color='tab:blue'
        )
        line_pred, = ax.plot(
            frames_test, Y_test_pred_sorted[:, k],
            linestyle='--',
            linewidth=0.8,
            label="predicted coordinates",
            #color='tab:orange'
        )

        ax.set_ylabel(coord_labels[k])

        # ---- apply manual ticks / limits for left axis if provided ----
        if coord_yticks_list[k] is not None:
            ax.set_yticks(coord_yticks_list[k])
        if coord_ylim_list[k] is not None:
            ax.set_ylim(*coord_ylim_list[k])

        ax.grid(True, which='both', linestyle='-', linewidth=0.5)

        # twin axis for absolute error
        ax_err = ax.twinx()
        line_err, = ax_err.plot(
            frames_test, errors_test[:, k],
            color='k',
            linewidth=0.4,
            label="absolute error"
        )
        ax_err.set_ylabel(error_labels[k])

        # ---- apply manual ticks / limits for right axis if provided ----
        if err_yticks_list[k] is not None:
            ax_err.set_yticks(err_yticks_list[k])
        if err_ylim_list[k] is not None:
            ax_err.set_ylim(*err_ylim_list[k])

        # legend on top subplot only
        if k == 0:
            lines = [line_ref, line_pred, line_err]
            labels = [
                "reference coordinates",
                "predicted coordinates",
                "absolute error"
            ]
            ax.legend(lines, labels, loc="upper right",ncol=3, frameon=True)

    axes[-1].set_xlabel("frames")
    plt.tight_layout(pad=0)
    plt.show()


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    train_svr()

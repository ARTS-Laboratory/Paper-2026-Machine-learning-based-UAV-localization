import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

from matplotlib import cm
from matplotlib.colors import Normalize

plt.close("all")


# ============================================================
# Gradient Boosting Regression with CV + Test Hyperparameter Analysis
# ============================================================
def train_gradient_boosting():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_excel("../data/filtered_Coordinates.xlsx")

    X = df[["LF_X", "F_Y", "RF_Z"]].values
    Y = df[["opt_X", "opt_Y", "opt_Z"]].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------
    # n_estimators_list = [ 300, 400, 500, 600, 700]
    # max_depth_list = [ 3, 4, 5, 6, 7]
    
    n_estimators_list = [400, 450, 500, 550]
    max_depth_list =    [3,4,5,6]

    # n_estimators_list = [ 700]
    # max_depth_list = [  5]

    cv_mse = np.zeros((len(max_depth_list), len(n_estimators_list)))
    test_mse = np.zeros_like(cv_mse)

    best_test_mse = np.inf
    best_test_idx = None
    best_test_model = None

    best_cv_mse = np.inf
    best_cv_idx = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------
    for i, max_depth in enumerate(max_depth_list):
        for j, n_estimators in enumerate(n_estimators_list):

            base_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                # learning_rate=0.1,
                # subsample=0.8,
                # max_features="sqrt",
                # loss="squared_error",
                random_state=42
            )

            model = MultiOutputRegressor(base_model)

            # ---- 5-fold CV MSE ----
            cv_errors = []
            for tr, va in kf.split(X_train):
                model.fit(X_train[tr], Y_train[tr])
                pred = model.predict(X_train[va])
                cv_errors.append(mean_squared_error(Y_train[va], pred))

            cv_mse[i, j] = np.mean(cv_errors)

            # ---- Test MSE ----
            model.fit(X_train, Y_train)
            pred_test = model.predict(X_test)
            test_mse[i, j] = mean_squared_error(Y_test, pred_test)

            # ---- Track best CV ----
            if cv_mse[i, j] < best_cv_mse:
                best_cv_mse = cv_mse[i, j]
                best_cv_idx = (i, j)

            # ---- Track best Test ----
            if test_mse[i, j] < best_test_mse:
                best_test_mse = test_mse[i, j]
                best_test_idx = (i, j)
                best_test_model = model

            print(f"Depth={max_depth}, Trees={n_estimators} | "
                  f"CV MSE={cv_mse[i,j]:.2f}, Test MSE={test_mse[i,j]:.2f}")

    # --------------------------------------------------------
    # Report best parameters
    # --------------------------------------------------------
    best_cv_depth = max_depth_list[best_cv_idx[0]]
    best_cv_trees = n_estimators_list[best_cv_idx[1]]

    best_test_depth = max_depth_list[best_test_idx[0]]
    best_test_trees = n_estimators_list[best_test_idx[1]]

    print("\n================ BEST CV MODEL ================")
    print(f"Best Max Depth (CV) : {best_cv_depth}")
    print(f"Best Trees (CV)     : {best_cv_trees}")
    print(f"Best CV MSE         : {best_cv_mse:.2f}")

    print("\n=============== BEST TEST MODEL ===============")
    print(f"Best Max Depth (Test) : {best_test_depth}")
    print(f"Best Trees (Test)     : {best_test_trees}")
    print(f"Best Test MSE         : {best_test_mse:.2f}")

    # --------------------------------------------------------
    # 3D line plot: Actual vs Predicted trajectory
    # --------------------------------------------------------
    Y_pred = best_test_model.predict(X)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2],
            color="green", linewidth=2, label="Actual OptiTrack")

    ax.plot(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2],
            color="blue", linestyle="--", linewidth=2,
            label="Predicted OptiTrack")

    ax.set_title("Gradient Boosting: Actual vs Predicted OptiTrack Trajectory")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Hyperparameter surface plots
    # --------------------------------------------------------
    D, T = np.meshgrid(max_depth_list, n_estimators_list, indexing="ij")

    def plot_surface(Z, title, zlabel, cmap):

        vmin = np.percentile(Z, 5)
        vmax = np.percentile(Z, 95)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(Z))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            D, T, Z,
            facecolors=colors,
            linewidth=0,
            antialiased=True,
            shade=False
        )

        # ---- Mark CV minimum ----
        ax.scatter(
            D[best_cv_idx[0], best_cv_idx[1]],
            T[best_cv_idx[0], best_cv_idx[1]],
            Z[best_cv_idx[0], best_cv_idx[1]],
            color="red",
            s=120,
            marker="*",
            label="CV Minimum"
        )

        # ---- Mark Test minimum ----
        ax.scatter(
            D[best_test_idx[0], best_test_idx[1]],
            T[best_test_idx[0], best_test_idx[1]],
            Z[best_test_idx[0], best_test_idx[1]],
            color="black",
            s=80,
            marker="x",
            label="Test Minimum"
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(Z)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=zlabel)

        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Number of Trees")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    plot_surface(
        cv_mse,
        "5-Fold CV MSE Hyperparameter Surface (Gradient Boosting)",
        "5-Fold CV MSE",
        cm.viridis
    )

    plot_surface(
        test_mse,
        "Test MSE Hyperparameter Surface (Gradient Boosting)",
        "Test MSE",
        cm.plasma
    )


# ============================================================
# Run
# ============================================================
train_gradient_boosting()

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


setup_done = False


def setup_fonts():
    global setup_done
    if setup_done:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    setup_done = True


setup_fonts()


def plot_iso_loss_contours(
    mu_rec_range: np.ndarray,
    token_range: np.ndarray,
    loss_surface: np.ndarray,
    efficient_frontier: Optional[List[Tuple[float, float, float]]] = None,
    optimal_mu: Optional[float] = None,
    optimal_D: Optional[float] = None,
    iso_flop_lines: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    fig, ax = plt.subplots(figsize=(10, 7))

    MU, D = np.meshgrid(mu_rec_range, token_range)
    loss_clipped = np.clip(loss_surface.T, np.nanmin(loss_surface) * 0.99, np.nanmax(loss_surface) * 1.01)

    levels = np.linspace(np.nanmin(loss_clipped), np.nanmax(loss_clipped), 20)
    contour = ax.contourf(MU, D, loss_clipped, levels=levels, cmap="viridis_r", alpha=0.9)
    ax.contour(MU, D, loss_clipped, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
    cbar = fig.colorbar(contour, ax=ax, label="Validation Loss", shrink=0.85)

    if efficient_frontier is not None and len(efficient_frontier) > 0:
        ef_mu = [p[0] for p in efficient_frontier]
        ef_D = [p[1] for p in efficient_frontier]
        ef_loss = [p[2] for p in efficient_frontier]
        scatter = ax.scatter(
            ef_mu, ef_D, c=ef_loss, cmap="plasma", s=80,
            edgecolors="white", linewidths=1.5, zorder=5,
            vmin=np.nanmin(loss_clipped), vmax=np.nanmax(loss_clipped),
        )
        ax.plot(ef_mu, ef_D, "w--", linewidth=1.5, alpha=0.8, zorder=4, label="Efficient frontier")

    if optimal_mu is not None and optimal_D is not None:
        ax.scatter([optimal_mu], [optimal_D], c="red", s=200, marker="*",
                   edgecolors="black", linewidths=1.0, zorder=6, label="Global optimum")

    if iso_flop_lines is not None:
        colors_iso = plt.cm.Set2(np.linspace(0, 1, max(len(iso_flop_lines), 1)))
        for idx, (mu_line, D_line) in enumerate(iso_flop_lines):
            ax.plot(
                [mu_line] * 2,
                [min(token_range) * 0.8, max(token_range) * 1.2],
                color=colors_iso[idx % len(colors_iso)],
                linestyle=":", linewidth=1.0, alpha=0.6,
            )
            ax.plot(
                [min(mu_rec_range) * 0.8, max(mu_rec_range) * 1.2],
                [D_line] * 2,
                color=colors_iso[idx % len(colors_iso)],
                linestyle=":", linewidth=1.0, alpha=0.6,
            )

    ax.set_xlabel(r"$\mu_{\mathrm{rec}}$")
    ax.set_ylabel(r"Tokens $D$")
    ax.set_title("Iso-Loss Contours over Recurrence and Compute Budget")
    ax.legend(loc="upper right", framealpha=0.9)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_efficient_frontier(
    flop_budgets: np.ndarray,
    optimal_mu_rec: np.ndarray,
    optimal_tokens: np.ndarray,
    frontier_losses: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sort_idx = np.argsort(flop_budgets)
    flop_budgets = np.asarray(flop_budgets)[sort_idx]
    optimal_mu_rec = np.asarray(optimal_mu_rec)[sort_idx]
    optimal_tokens = np.asarray(optimal_tokens)[sort_idx]
    frontier_losses = np.asarray(frontier_losses)[sort_idx]

    log_flop = np.log10(flop_budgets)
    log_mu = np.log10(optimal_mu_rec)
    log_D = np.log10(optimal_tokens)

    coeffs_mu = np.polyfit(log_flop, log_mu, 1)
    coeffs_D = np.polyfit(log_flop, log_D, 1)

    ax1.scatter(log_flop, log_mu, c=frontier_losses, cmap="viridis_r", s=80, edgecolors="black", linewidths=0.5, zorder=3)
    fit_line_mu = coeffs_mu[0] * log_flop + coeffs_mu[1]
    ax1.plot(log_flop, fit_line_mu, "r--", linewidth=2, label=rf"$\alpha = {coeffs_mu[0]:.2f}$")
    ax1.set_xlabel(r"$\log_{10}(C)$ [FLOPs]")
    ax1.set_ylabel(r"$\log_{10}(\mu^*_{\mathrm{rec}})$")
    ax1.set_title(r"Optimal $\mu_{\mathrm{rec}}$ vs FLOP Budget")
    ax1.legend()

    scatter2 = ax2.scatter(log_flop, log_D, c=frontier_losses, cmap="viridis_r", s=80, edgecolors="black", linewidths=0.5, zorder=3)
    fit_line_D = coeffs_D[0] * log_flop + coeffs_D[1]
    ax2.plot(log_flop, fit_line_D, "r--", linewidth=2, label=rf"$\beta = {coeffs_D[0]:.2f}$")
    ax2.set_xlabel(r"$\log_{10}(C)$ [FLOPs]")
    ax2.set_ylabel(r"$\log_{10}(D^*)$ [Tokens]")
    ax2.set_title(r"Optimal Tokens vs FLOP Budget")
    ax2.legend()
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="Loss", shrink=0.85)

    fig.suptitle("Chinchilla-Optimal Allocation for Looped LMs", fontsize=15, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_power_law(
    x_values: np.ndarray,
    y_values: np.ndarray,
    fit_exponent: float,
    fit_intercept: float,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = np.asarray(x_values, dtype=np.float64)
    y_vals = np.asarray(y_values, dtype=np.float64)
    valid = (x_vals > 0) & (y_vals > 0)
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]

    log_x = np.log10(x_vals)
    log_y = np.log10(y_vals)

    ax.scatter(log_x, log_y, c="steelblue", s=80, edgecolors="black", linewidths=0.5, zorder=3, label="Data")

    x_fit = np.linspace(log_x.min(), log_x.max(), 200)
    log_intercept_10 = fit_intercept / np.log(10)
    y_fit = fit_exponent * x_fit + log_intercept_10
    ax.plot(x_fit, y_fit, "r--", linewidth=2, label=rf"$\alpha = {fit_exponent:.3f}$")

    residuals = log_y - (fit_exponent * log_x + log_intercept_10)
    r_squared = 1.0 - np.sum(residuals ** 2) / np.sum((log_y - np.mean(log_y)) ** 2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ($R^2 = {r_squared:.4f}$)")
    ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_saturation_curves(
    T_values: np.ndarray,
    loss_curves: Dict[float, List[float]],
    mu_rec_values: Optional[List[float]] = None,
    fit_results: Optional[Dict[float, Dict]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    T_arr = np.asarray(T_values, dtype=np.float64)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loss_curves), 1)))

    for idx, (mu_rec, losses) in enumerate(sorted(loss_curves.items())):
        losses_arr = np.asarray(losses, dtype=np.float64)
        T_plot = T_arr[: len(losses_arr)]
        label = rf"$\mu_{{\mathrm{{rec}}}}={mu_rec:.0f}$"
        ax1.plot(T_plot, losses_arr, "o-", color=colors[idx % len(colors)],
                 markersize=4, linewidth=1.5, label=label, alpha=0.8)

        if fit_results is not None and mu_rec in fit_results:
            fit = fit_results[mu_rec]
            if "exponential" in fit and fit["exponential"]["params"] is not None:
                p = fit["exponential"]["params"]
                L_inf, Z, z = p["L_inf"], p["Z"], p["z"]
                T_fine = np.linspace(T_plot.min(), T_plot.max(), 200)
                L_fit = L_inf + Z * np.exp(-z * T_fine)
                ax1.plot(T_fine, L_fit, "--", color=colors[idx % len(colors)],
                         linewidth=2, alpha=0.9)
                ax1.axhline(y=L_inf, color=colors[idx % len(colors)],
                             linestyle=":", alpha=0.4, linewidth=1)

    ax1.set_xlabel("Test-time steps $T$")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Test-Time Saturation Curves")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    if fit_results is not None:
        mu_recs_fitted = sorted(fit_results.keys())
        huber_ids = []
        huber_oods = []
        floor_errors = []
        labels_bar = []
        for mu_rec in mu_recs_fitted:
            fit = fit_results[mu_rec]
            if "exponential" in fit:
                huber_ids.append(fit["exponential"]["huber_id"])
                huber_oods.append(fit["exponential"]["huber_ood"])
                fe = fit["exponential"].get("floor_error_pct")
                floor_errors.append(fe if fe is not None else 0.0)
                labels_bar.append(rf"$\mu_{{\mathrm{{rec}}}}={mu_rec:.0f}$")

        x_pos = np.arange(len(labels_bar))
        width = 0.35
        if huber_ids:
            bars1 = ax2.bar(x_pos - width / 2, huber_ids, width, label="In-distribution", color="steelblue", alpha=0.8)
        if huber_oods:
            finite_oods = [h if np.isfinite(h) else 0 for h in huber_oods]
            bars2 = ax2.bar(x_pos + width / 2, finite_oods, width, label="Extrapolation", color="coral", alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels_bar, rotation=45, ha="right", fontsize=9)
        ax2.set_ylabel("Huber Loss")
        ax2.set_title("Fit Quality: In-Distribution vs Extrapolation")
        ax2.legend()
        ax2.set_yscale("symlog", linthresh=1e-6)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.axis("off")

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_training_scaling(
    param_values: np.ndarray,
    token_values: np.ndarray,
    loss_grid: np.ndarray,
    fit_params: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()

    log_N = np.log10(np.maximum(param_values, 1.0))
    log_D = np.log10(np.maximum(token_values, 1.0))
    log_loss = np.log10(np.maximum(loss_grid, 1e-12))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    LOG_N, LOG_D = np.meshgrid(log_N, log_D)
    surf = ax.plot_surface(LOG_N, LOG_D, log_loss.T, cmap="viridis_r", alpha=0.85, edgecolor="none")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, label=r"$\log_{10}$(Loss)", pad=0.1)

    if fit_params is not None:
        E = fit_params.get("E", 0)
        X = fit_params.get("X", 1)
        x = fit_params.get("x", 0.5)
        Y = fit_params.get("Y", 1)
        y = fit_params.get("y", 0.5)
        N_fine = np.logspace(log_N.min(), log_N.max(), 50)
        D_fine = np.logspace(log_D.min(), log_D.max(), 50)
        loss_fit = E + X * N_fine[:, None] ** (-x) + Y * D_fine[None, :] ** (-y)
        log_N_fine = np.log10(N_fine)
        log_D_fine = np.log10(D_fine)
        LOG_NF, LOG_DF = np.meshgrid(log_N_fine, log_D_fine)
        ax.plot_wireframe(LOG_NF, LOG_DF, np.log10(np.maximum(loss_fit.T, 1e-12)),
                          color="red", linewidth=0.5, alpha=0.5, rstride=5, cstride=5)

    ax.set_xlabel(r"$\log_{10}(N_{\mathrm{eff}})$")
    ax.set_ylabel(r"$\log_{10}(D)$")
    ax.set_zlabel(r"$\log_{10}(\mathrm{Loss})$")
    ax.set_title("Training Scaling Law Surface")
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_unified_law_predictions(
    T_values: np.ndarray,
    actual_losses: Dict[float, List[float]],
    predicted_losses: Dict[float, List[float]],
    mu_rec_values: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    T_arr = np.asarray(T_values, dtype=np.float64)
    mu_recs = sorted(actual_losses.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(mu_recs), 1)))

    for idx, mu_rec in enumerate(mu_recs):
        act = np.asarray(actual_losses[mu_rec], dtype=np.float64)
        pred = np.asarray(predicted_losses[mu_rec], dtype=np.float64)
        T_plot = T_arr[: min(len(act), len(pred))]
        act_plot = act[: len(T_plot)]
        pred_plot = pred[: len(T_plot)]

        label = rf"$\mu_{{\mathrm{{rec}}}}={mu_rec:.0f}$"
        ax1.plot(T_plot, act_plot, "o-", color=colors[idx % len(colors)],
                 markersize=4, linewidth=1.5, alpha=0.8, label=label)
        ax1.plot(T_plot, pred_plot, "--", color=colors[idx % len(colors)],
                 linewidth=2, alpha=0.9)

    ax1.set_xlabel("Test-time steps $T$")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Unified Law: Predicted vs Actual")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    all_actual = []
    all_predicted = []
    for mu_rec in mu_recs:
        act = np.asarray(actual_losses[mu_rec], dtype=np.float64)
        pred = np.asarray(predicted_losses[mu_rec], dtype=np.float64)
        n = min(len(act), len(pred))
        all_actual.extend(act[:n].tolist())
        all_predicted.extend(pred[:n].tolist())

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    min_val = min(all_actual.min(), all_predicted.min())
    max_val = max(all_actual.max(), all_predicted.max())
    margin = (max_val - min_val) * 0.05

    ax2.scatter(all_actual, all_predicted, c="steelblue", s=30, alpha=0.6, edgecolors="none")
    ax2.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
             "r--", linewidth=2, label="Perfect prediction")

    mape = np.mean(np.abs((all_predicted - all_actual) / all_actual)) * 100
    ax2.text(0.05, 0.95, f"MAPE = {mape:.3f}%",
             transform=ax2.transAxes, fontsize=12, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax2.set_xlabel("Actual Loss")
    ax2.set_ylabel("Predicted Loss")
    ax2.set_title("Prediction Accuracy")
    ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_stability_diagnostics(
    steps: np.ndarray,
    spectral_radii: np.ndarray,
    state_norms: np.ndarray,
    residual_norms: np.ndarray,
    lr_values: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()
    n_plots = 3 + (1 if lr_values is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    steps = np.asarray(steps, dtype=np.float64)
    spectral_radii = np.asarray(spectral_radii, dtype=np.float64)
    state_norms = np.asarray(state_norms, dtype=np.float64)
    residual_norms = np.asarray(residual_norms, dtype=np.float64)

    axes[0].plot(steps, spectral_radii, color="crimson", linewidth=1.5, alpha=0.8)
    axes[0].axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Stability threshold")
    axes[0].fill_between(steps, 1.0, spectral_radii,
                         where=spectral_radii > 1.0, color="red", alpha=0.15, label="Unstable region")
    axes[0].set_ylabel(r"Spectral radius $\rho(\mathbf{A})$")
    axes[0].set_title("Spectral Radius Over Training")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(steps, state_norms, color="steelblue", linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel(r"Hidden state norm $\|\mathbf{h}_t\|$")
    axes[1].set_title("Hidden State Norms Over Training")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(steps, residual_norms, color="forestgreen", linewidth=1.5, alpha=0.8)
    axes[2].set_ylabel(r"Residual norm $\|\mathbf{r}_t\|$")
    axes[2].set_title("Residual Stream Norms Over Training")
    axes[2].grid(True, alpha=0.3)

    if lr_values is not None:
        lr_values = np.asarray(lr_values, dtype=np.float64)
        axes[3].semilogy(steps, lr_values, color="darkorange", linewidth=1.5, alpha=0.8)
        axes[3].set_ylabel(r"Learning rate $\eta$")
        axes[3].set_title("Learning Rate Schedule")
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training step")
    fig.suptitle("Stability Diagnostics", fontsize=15, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def plot_ablation_results(
    ablation_names: List[str],
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_fonts()

    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    n_ablations = len(ablation_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_ablations)
    width = 0.8 / n_metrics
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for m_idx, metric_name in enumerate(metric_names):
        ax = axes[m_idx]
        values = [metrics_dict[metric_name].get(name, 0.0) for name in ablation_names]

        bars = ax.bar(x, values, width * n_metrics, color=colors[m_idx], alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, values):
            y_offset = val + max(values) * 0.01 if val >= 0 else val - max(values) * 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, y_offset,
                    f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(ablation_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Ablation Study Results", fontsize=15, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig


def save_all_scaling_plots(
    results_dir: str,
    all_results: Dict[str, Any],
) -> List[str]:
    os.makedirs(results_dir, exist_ok=True)
    saved_files = []

    if "iso_loss_data" in all_results:
        data = all_results["iso_loss_data"]
        if all(k in data for k in ("mu_rec_range", "token_range", "loss_surface")):
            path = os.path.join(results_dir, "iso_loss_contours.png")
            ef = data.get("efficient_frontier")
            omu = data.get("optimal_mu")
            oD = data.get("optimal_D")
            iso_lines = data.get("iso_flop_lines")
            plot_iso_loss_contours(
                data["mu_rec_range"], data["token_range"], data["loss_surface"],
                efficient_frontier=ef, optimal_mu=omu, optimal_D=oD,
                iso_flop_lines=iso_lines, save_path=path,
            )
            saved_files.append(path)

    if "efficient_frontier_data" in all_results:
        data = all_results["efficient_frontier_data"]
        if all(k in data for k in ("flop_budgets", "optimal_mu_rec", "optimal_tokens", "frontier_losses")):
            path = os.path.join(results_dir, "efficient_frontier.png")
            plot_efficient_frontier(
                data["flop_budgets"], data["optimal_mu_rec"],
                data["optimal_tokens"], data["frontier_losses"],
                save_path=path,
            )
            saved_files.append(path)

    if "mu_rec_power_law" in all_results:
        data = all_results["mu_rec_power_law"]
        if all(k in data for k in ("flop_budgets", "optimal_mu_rec", "exponent", "intercept")):
            path = os.path.join(results_dir, "mu_rec_power_law.png")
            plot_power_law(
                data["flop_budgets"], data["optimal_mu_rec"],
                data["exponent"], data["intercept"],
                xlabel=r"$\log_{10}(C)$", ylabel=r"$\log_{10}(\mu^*_{\mathrm{rec}})$",
                title=r"$\mu^*_{\mathrm{rec}} \propto C^{\alpha}$",
                save_path=path,
            )
            saved_files.append(path)

    if "token_power_law" in all_results:
        data = all_results["token_power_law"]
        if all(k in data for k in ("flop_budgets", "optimal_tokens", "exponent", "intercept")):
            path = os.path.join(results_dir, "token_power_law.png")
            plot_power_law(
                data["flop_budgets"], data["optimal_tokens"],
                data["exponent"], data["intercept"],
                xlabel=r"$\log_{10}(C)$", ylabel=r"$\log_{10}(D^*)$",
                title=r"$D^* \propto C^{\beta}$",
                save_path=path,
            )
            saved_files.append(path)

    if "saturation_data" in all_results:
        data = all_results["saturation_data"]
        if "T_values" in data and "loss_curves" in data:
            path = os.path.join(results_dir, "saturation_curves.png")
            plot_saturation_curves(
                data["T_values"], data["loss_curves"],
                mu_rec_values=data.get("mu_rec_values"),
                fit_results=data.get("fit_results"),
                save_path=path,
            )
            saved_files.append(path)

    if "training_scaling_data" in all_results:
        data = all_results["training_scaling_data"]
        if all(k in data for k in ("param_values", "token_values", "loss_grid")):
            path = os.path.join(results_dir, "training_scaling_surface.png")
            plot_training_scaling(
                data["param_values"], data["token_values"], data["loss_grid"],
                fit_params=data.get("fit_params"),
                save_path=path,
            )
            saved_files.append(path)

    if "unified_predictions" in all_results:
        data = all_results["unified_predictions"]
        if "T_values" in data and "actual_losses" in data and "predicted_losses" in data:
            path = os.path.join(results_dir, "unified_law_predictions.png")
            plot_unified_law_predictions(
                data["T_values"], data["actual_losses"], data["predicted_losses"],
                mu_rec_values=data.get("mu_rec_values"),
                save_path=path,
            )
            saved_files.append(path)

    if "stability_data" in all_results:
        data = all_results["stability_data"]
        if all(k in data for k in ("steps", "spectral_radii", "state_norms", "residual_norms")):
            path = os.path.join(results_dir, "stability_diagnostics.png")
            plot_stability_diagnostics(
                data["steps"], data["spectral_radii"],
                data["state_norms"], data["residual_norms"],
                lr_values=data.get("lr_values"),
                save_path=path,
            )
            saved_files.append(path)

    if "ablation_data" in all_results:
        data = all_results["ablation_data"]
        if "names" in data and "metrics" in data:
            path = os.path.join(results_dir, "ablation_results.png")
            plot_ablation_results(data["names"], data["metrics"], save_path=path)
            saved_files.append(path)

    return saved_files

from typing import Optional
import wandb

from ml_evaluation.interactive_confusion_matrix import get_interactive_confusion_matrix, \
    get_interactive_confusion_matrix_with_slider
from ml_evaluation.interactive_cumulative_gains_curve import get_interactive_cumulative_gains_curve
from ml_evaluation.interactive_ks_curve import get_interactive_ks_curve, get_interactive_ks_cumulative_plot, \
    get_interactive_ks_comprehensive_plot, get_interactive_ks_simple
from ml_evaluation.interactive_lift_chart import get_interactive_lift_chart, get_interactive_lift_chart_two_color, \
    get_interactive_lift_chart_gradient
from ml_evaluation.interactive_pr_curve import get_interactive_pr_curve
from ml_evaluation.interactive_roc_curve import get_interactive_roc_curve


def clean():
    api = wandb.Api()
    runs = api.runs("vslovik/france-hvac")
    for run in runs:
        run.delete()


def init_wandb(project_name: str = "france-hvac", run_name: Optional[str] = None, run_id: Optional[str] = None) -> str:
    settings = dict()
    settings['project'] = project_name
    if run_id is not None:
        settings['id'] = run_id
        settings['resume'] = 'allow'
    if run_name is not None:
        settings['name'] = run_name
    run = wandb.init(**settings)
    return run.id


def log_roc_curve(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    fig = get_interactive_roc_curve(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "roc_curve"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_pr_curve(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    fig = get_interactive_pr_curve(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "pr_curve"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_ks_curve(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    #fig = get_interactive_ks_curve(y_test, y_pred_proba, model_name)
    #fig = get_interactive_ks_cumulative_plot(y_test, y_pred_proba, model_name)
    #fig = get_interactive_ks_comprehensive_plot(y_test, y_pred_proba, model_name)
    fig = get_interactive_ks_simple(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "ks_curve"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_lift_chart(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    #fig = get_interactive_lift_chart(y_test, y_pred_proba, model_name)
    #fig = get_interactive_lift_chart_two_color(y_test, y_pred_proba, model_name)
    fig = get_interactive_lift_chart_gradient(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "lift_chart"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_cumulative_gains_curve(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    fig = get_interactive_cumulative_gains_curve(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "cumulative_gains_curve"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_confusion_matrix(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    #fig = get_interactive_confusion_matrix(y_test, y_pred_proba, model_name)
    fig = get_interactive_confusion_matrix_with_slider(y_test, y_pred_proba, model_name)

    init_wandb(run_id=wandb_run_id)
    plot_name = "confusion_matrix"
    path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
    wandb.log({
        path: fig,
    })

    return fig


def log_all_plots(y_test, y_pred_proba, wandb_run_id, model_name: str = ""):
    plots = {
        'roc_curve': get_interactive_roc_curve(y_test, y_pred_proba, model_name),
        'pr_curve': get_interactive_pr_curve(y_test, y_pred_proba, model_name),
        'ks_simple': get_interactive_ks_simple(y_test, y_pred_proba, model_name),
        'lift_chart': get_interactive_lift_chart_gradient(y_test, y_pred_proba, model_name),
        'cumulative_gains_curve': get_interactive_cumulative_gains_curve(y_test, y_pred_proba, model_name),
        'confusion_matrix': get_interactive_confusion_matrix_with_slider(y_test, y_pred_proba, model_name)
    }
    init_wandb(run_id=wandb_run_id)
    for plot_name, fig in plots.items():
        path = f"_plots/{plot_name}/{model_name}" if model_name else f"_plots/{plot_name}"
        wandb.log({
            path: fig,
        })
    return plots


def log_plot(fig, wandb_run_id, plot_name: str, model_name: str = ""):
    init_wandb(run_id=wandb_run_id)
    wandb.log({
        f"_plots/{plot_name}": fig,
    })
    return fig


def log_plot_as_image(fig, wandb_run_id, plot_name: str, model_name: str = ""):
    init_wandb(run_id=wandb_run_id)
    wandb.log({
        f"_plots/{plot_name}": wandb.Image(fig),
    })
    return fig
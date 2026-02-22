import wandb
import ipywidgets as widgets
from IPython.display import display


class Widget:
    OPTIONS = {}

    def __init__(self, compute_func, selected_ids, log_to_wandb=False):
        self.compute_func = compute_func
        self.selected_ids = selected_ids
        self.log_to_wandb = log_to_wandb

    def get_make_fig(self):
        return lambda data, key: None

    def get_dropdown(self):
        return widgets.Dropdown(
            options=list(self.OPTIONS.keys()),
            value="Actuel",
            description='Scénario :',
            layout={'width': '380px'}
        )

    def show(self):
        make_fig = self.get_make_fig()
        dropdown = self.get_dropdown()
        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                key = dropdown.value
                scenario = self.OPTIONS[key]['scenario']
                data = self.compute_func(family=scenario)
                fig = make_fig(data, key)
                display(fig)

        dropdown.observe(update, names='value')

        # Show UI
        ui = widgets.VBox([
            widgets.HBox([dropdown]),
            output
        ])

        update()  # initial plot
        display(ui)

        if self.log_to_wandb:
            table = wandb.Table(columns=["Scenario", "Plot"])
            for i, scen in enumerate(self.OPTIONS):
                if i == 0:
                    continue
                fam = self.OPTIONS[scen]["scenario"]
                data = self.compute_func(family=fam)
                fig = make_fig(data, scen)
                fig.show()
                html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                table.add_data(scen, wandb.Html(html))
            wandb.log({"follow_up_comparison": table})
        else:
            for i, scen in enumerate(self.OPTIONS):
                if i == 0:
                    continue
                fam = self.OPTIONS[scen]["scenario"]
                data = self.compute_func(family=fam)
                fig = make_fig(data, scen)
                fig.show()

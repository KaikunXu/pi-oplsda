# src/piopls/oplsda_plotting.py
"""Visualization suite for OPLS-DA models.

This module consumes the Pandas DataFrames exported by the OPLSDA model 
to generate publication-ready diagnostic plots, including Model Overview, 
Score Plot, Permutation Test Plot, S-Plot, and VIP Bar Plot.
"""

import numpy as np
import pandas as pd
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.transforms import Affine2D
import seaborn as sns


class OPLSDA_Visualizer:
    """Visualization suite generating the 5 standard diagnostic plots.

    Generates (A) Model Overview, (B) X-Score Plot, (C) Permutation Test, 
    (D) S-Plot, and (E) VIP Bar Plot.

    Attributes:
        model (OPLSDA): The fitted OPLSDA model instance.
        feature_names (list): List of feature names.
        sample_names (list): List of sample names.
        class_names (dict): Mapping of encoded integers to class names.
        y_groups (list): Class assignments for visualization grouping.
        vip_threshold (float): Threshold to highlight features in S-Plot.
        top_n_vip (int): Number of top features for VIP bar plot.
        palette (dict): Color mapping for categorical variables.
    """

    def __init__(
        self, 
        model, 
        X=None, 
        y=None, 
        feature_names=None, 
        sample_names=None, 
        vip_threshold=1.0, 
        top_n_vip=25, 
        custom_palette=None
    ):
        """Initializes the visualizer with the model and rendering settings.

        Args:
            model (OPLSDA): The fitted OPLSDA model.
            X (array-like, optional): Feature matrix. Defaults to None.
            y (array-like, optional): Target vector. Defaults to None.
            feature_names (list, optional): Names of features. Defaults to None.
            sample_names (list, optional): Names of samples. Defaults to None.
            vip_threshold (float, optional): VIP cutoff for S-plot styling. 
                Defaults to 1.0.
            top_n_vip (int, optional): Max features for VIP bar plot. 
                Defaults to 25.
            custom_palette (dict, optional): Custom color mapping for classes. 
                Defaults to None.
        """
        # ==========================================
        # Global Matplotlib & Seaborn Configuration
        # ==========================================
        # Ensure high-quality vector export
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
        
        # Force white style to ensure background consistency
        sns.set_style("ticks")
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        
        # Data and Attribute Loading        
        self.model = model
        self.feature_names = feature_names
        self.sample_names = sample_names
        
        # Configure class names and groups for plotting legends
        if (hasattr(model, 'label_encoder') and 
                len(model.label_encoder.classes_) > 0):
            self.class_names = {
                i: name for i, name in enumerate(model.label_encoder.classes_)
            }
            if y is not None:
                self.y_groups = y
            else:
                self.y_groups = [self.class_names[0]] * len(model.t_pred_)
        else:
            self.class_names = {0: 'Class 0', 1: 'Class 1'}
            if y is not None:
                median_y = np.median(y)
                self.y_groups = [
                    'Class 1' if v > median_y else 'Class 0' for v in y
                ]
            else:
                self.y_groups = ['Class 0'] * len(model.t_pred_)

        self.vip_threshold = vip_threshold
        self.top_n_vip = top_n_vip
        
        # Assign color palette dynamically
        if custom_palette is not None:
            self.palette = custom_palette
        else:
            keys = list(set(self.y_groups))
            if len(keys) >= 2:
                self.palette = {keys[0]: 'tab:blue', keys[1]: 'tab:red'}
            else:
                self.palette = {keys[0]: 'tab:blue'}

    def _draw_confidence_ellipse(self, x, y, ax, n_std=2.0, **kwargs):
        """Calculates and draws a confidence ellipse representing covariance.

        Args:
            x (array-like): X-axis coordinates.
            y (array-like): Y-axis coordinates.
            ax (matplotlib.axes.Axes): Target axes object.
            n_std (float, optional): Number of standard deviations for 
                the ellipse radius. Defaults to 2.0.
            **kwargs: Additional styling arguments for the Ellipse patch.
        """
        if x.size != y.size or x.size < 3: 
            return
            
        cov = np.cov(x, y)
        if cov[0, 0] == 0 or cov[1, 1] == 0: 
            return
            
        # Pearson correlation coefficient
        pearson = np.clip(
            cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), -1.0, 1.0
        )
        
        ellipse = Ellipse(
            (0, 0), 
            width=np.sqrt(1 + pearson) * 2, 
            height=np.sqrt(1 - pearson) * 2, 
            **kwargs
        )
        
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        
        transf = Affine2D().rotate_deg(45).scale(scale_x, scale_y)
        transf = transf.translate(np.mean(x), np.mean(y))
        
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    def _format_ax(self, ax, xlabel, ylabel, title):
        """Applies consistent aesthetic formatting to an axes object.

        Args:
            ax (matplotlib.axes.Axes): Target axes object.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title for the plot.
        """
        ax.set_xlabel(xlabel, fontsize=11, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='normal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=11)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    def _add_legend(self, ax, title="", loc="best", ncol=1):
        """Applies consistent aesthetic formatting to a legend.

        Args:
            ax (matplotlib.axes.Axes): Target axes object.
            title (str, optional): Title for the legend. Defaults to "".
            loc (str, optional): Matplotlib location string. Defaults to "best".
            ncol (int, optional): Number of columns in the legend. Defaults to 1.
        """
        ax.legend(
            title=title, ncol=ncol, frameon=True, shadow=True, 
            fontsize=10, borderpad=0.4, facecolor="white", loc=loc
        )

    def plot_all(
        self,
        perm_results = None,
        save_path = None,
        wrap_width = 20,
        figsize = (15, 10),
        return_fig = False
    ):
        """Generates and displays diagnostic plots using patchworklib.

        Combines the Model Overview, X-Score Plot, Permutation Test, S-Plot, 
        and VIP Bar Plot using patchworklib for perfect coordinate alignment.

        Args:
            perm_results (dict, optional): Dictionary containing permutation 
                results to enable the permutation test subplot. Defaults to None.
            save_path (str, optional): File path to save the generated figure. 
                Defaults to None.
            wrap_width (int, optional): Max character width for VIP y-labels 
                before wrapping. Defaults to 20.
            figsize (tuple, optional): Width, height in inches for the figure. 
                Defaults to (15, 10).
            return_fig (bool, optional): If True, returns the pw.Brick object. 
                Defaults to False to prevent double-plotting in Jupyter Notebook
                environments.
        """
        import patchworklib as pw
        
        total_width,total_height = figsize
        
        # Calculate proportional widths
        total_parts = 1 + 1 + 0.8
        w_col1 = total_width * (1 / total_parts)
        w_col2 = total_width * (1 / total_parts)
        w_col3 = total_width * (0.8 / total_parts)
        
        # Calculate height for the 2x2 grid blocks (left side)
        h_half = total_height / 2
        
        # Initialize pw.Brick objects with strictly assigned dimensions
        ax_ov = pw.Brick(figsize=(w_col1, h_half))
        ax_sc = pw.Brick(figsize=(w_col2, h_half))
        ax_pm = pw.Brick(figsize=(w_col1, h_half))
        ax_sp = pw.Brick(figsize=(w_col2, h_half))
        ax_vp = pw.Brick(figsize=(w_col3, total_height))

        # Render individual plots onto their respective bricks
        self.plot_model_overview(ax=ax_ov)
        self.plot_score(ax=ax_sc)
        self.plot_splot(ax=ax_sp)
        self.plot_vip_bar(ax=ax_vp, wrap_width=wrap_width)
        
        if perm_results: 
            self.plot_permutations(perm_results, ax=ax_pm)
        else: 
            ax_pm.axis('off')

        # Construct the layout using patchworklib algebra
        final_fig = ((ax_ov|ax_sc)/(ax_pm|ax_sp))|ax_vp
        
        if save_path:
            final_fig.savefig(save_path, dpi=300, bbox_inches='tight')

        if not return_fig:
            return None
        
        # Disconnect all active figures from the Matplotlib pyplot state
        # to prevent Jupyter's inline backend from auto-displaying them.
        # The 'final_fig' object retains memory references to render correctly.
        plt.close('all')
        
        return final_fig

    def plot_model_overview(self, ax=None):
        """Plots a grouped bar chart showing R2Y and Q2 for each component.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes object. 
                If None, creates a new figure.
        """
        df_summary = self.model.get_summary_df()
        labels = df_summary['Component'].tolist()
        r2y_vals = df_summary['R2Y'].tolist()
        q2_vals  = df_summary['Q2'].tolist()
        
        if ax is None: 
            fig_w = max(4.5, len(labels) * 1.5)
            _, ax = plt.subplots(figsize=(fig_w, 3))
            
        x = np.arange(len(labels))
        width = 0.4

        ax.bar(
            x - width/2, r2y_vals, width, label='R2Y', 
            color='#6BAED6', edgecolor='k', linewidth=0.5
        )
        ax.bar(
            x + width/2, q2_vals, width, label='Q2',  
            color='#2171B5', edgecolor='k', linewidth=0.5
        )

        def add_labels(vals, offset):
            for i, v in enumerate(vals):
                if v == 0: 
                    continue
                y_pos = v + 0.02 if v >= 0 else v - 0.05
                v_align = 'bottom' if v >= 0 else 'top'
                ax.text(
                    i + offset, y_pos, f'{v:.3f}', 
                    ha='center', va=v_align, fontsize=9
                )

        add_labels(r2y_vals, -width/2)
        add_labels(q2_vals, width/2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        max_val = max([max(r2y_vals), max(q2_vals)])
        min_val = min([min(q2_vals), 0])
        y_lim_min = min_val - 0.1 if min_val < 0 else 0
        
        ax.set_ylim(y_lim_min, min(1.2, max_val * 1.35))
        
        if min_val < 0: 
            ax.axhline(0, color='k', linewidth=0.8)
        
        self._format_ax(ax, "", "", "Model Overview")
        self._add_legend(ax, ncol=2, loc='upper left')

    def plot_score(self, ax=None):
        """Plots the sample score scatter plot.

        Visualizes the Predictive vs 1st Orthogonal T-score and automatically 
        maps groups using 'True_Class' and fits confidence ellipses.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes object. 
                If None, creates a new figure.
        """
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_scores = self.model.get_scores_df(
            sample_names=self.sample_names, y_true=self.y_groups
        )
        df_summary = self.model.get_summary_df()
        
        t1 = df_scores['t_pred']
        p_row = df_summary.loc[df_summary['Component'] == 'p1', 'R2X']
        p_explan = p_row.values[0] * 100
        
        # Handle cases with 0 orthogonal components
        if 't_ortho_1' in df_scores.columns:
            o1 = df_scores['t_ortho_1']
            o_row = df_summary.loc[df_summary['Component'] == 'o1', 'R2X']
            o_explan = o_row.values[0] * 100
        else:
            o1 = np.zeros_like(t1)
            o_explan = 0.0
            
        df_scores['t_ortho_1_plot'] = o1
            
        sns.scatterplot(
            data=df_scores, x='t_pred', y='t_ortho_1_plot', hue='True_Class', 
            palette=self.palette, edgecolor='k', linewidth=0.5, s=40, ax=ax
        )
        
        # Draw confidence ellipses for each group
        for cls_name in np.unique(df_scores['True_Class']):
            sub = df_scores[df_scores['True_Class'] == cls_name]
            if cls_name in self.palette:
                color = self.palette[cls_name]
                self._draw_confidence_ellipse(
                    sub['t_pred'], sub['t_ortho_1_plot'], ax, n_std=2.0, 
                    facecolor=color, edgecolor=color, alpha=0.3, linewidth=1
                )

        x_label = f"T Score [1]\n({p_explan:.1f} %)"
        y_label = f"Orthogonal T Score [1]\n({o_explan:.1f} %)"
        self._format_ax(ax, x_label, y_label, "X-Score Plot")
        self._add_legend(ax, ncol=2, loc='best')

    def plot_permutations(self, perm_results, ax=None):
        """Plots the distribution of R2Y and Q2 from the permutation test.

        Args:
            perm_results (dict): Dictionary returned from permutation_test().
            ax (matplotlib.axes.Axes, optional): Target axes object. 
                If None, creates a new figure.
        """
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
            
        perms_r2y = perm_results.get('perms_R2Y', [])
        perms_q2 = perm_results.get('perms_Q2', [])
        real_r2y = perm_results.get('orig_R2Y', 0)
        real_q2 = perm_results.get('orig_Q2', 0)
        
        if len(perms_r2y) > 0:
            arr_concat = np.concatenate([perms_r2y, perms_q2])
            bins = np.histogram_bin_edges(arr_concat, bins=30)
            
            ax.hist(
                perms_r2y, bins=bins, color='tab:red', 
                edgecolor='k', linewidth=0.5, label='Perm R2Y'
            )
            ax.hist(
                perms_q2, bins=bins, color='tab:blue', 
                edgecolor='k', linewidth=0.5, label='Perm Q2'
            )
            
        maxY = ax.get_ylim()[1]
        xmin, xmax = ax.get_xlim()
        needed_xmax = max(real_r2y, real_q2, xmax)
        ax.set_xlim(xmin, needed_xmax + (needed_xmax - xmin) * 0.15)
        
        # Mark real model performances with vertical lines
        ax.axvline(
            real_r2y, color='tab:red', linestyle='--', linewidth=1.0, zorder=0
        )
        ax.axvline(
            real_q2, color='tab:blue', linestyle='--', linewidth=1.0, zorder=0
        )

        # Style configurations for annotations
        arrow = dict(
            arrowstyle="-|>", connectionstyle="arc3, rad=-0.15", 
            color="k", linewidth=1.0
        )
        bbox = dict(boxstyle="round", facecolor="white", edgecolor="k", pad=0.2)
        
        # Annotate actual R2Y and Q2 with p-values
        r2y_p = perm_results.get('p_R2Y', 0)
        ax.annotate(
            f"R2Y: {real_r2y:.3f}\nP = {r2y_p:.3f}", 
            xy=(real_r2y, 0), xytext=(real_r2y, maxY * 0.75),
            ha="center", va="center", fontsize=8, bbox=bbox, arrowprops=arrow
        )
        
        q2_p = perm_results.get('p_Q2', 0)
        ax.annotate(
            f"Q2: {real_q2:.3f}\nP = {q2_p:.3f}", 
            xy=(real_q2, 0), xytext=(real_q2, maxY * 0.5),
            ha="center", va="center", fontsize=8, bbox=bbox, arrowprops=arrow
        )

        self._format_ax(ax, "Permutations", "Frequency", "Permutation Test")
        self._add_legend(ax, ncol=1, loc='best')

    def plot_splot(self, ax=None):
        """Plots the S-Plot (Covariance vs Correlation) highlighting features.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes object. 
                If None, creates a new figure.
        """
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_feat = self.model.get_features_df(feature_names=self.feature_names)
        
        cat_high = f'VIP>={self.vip_threshold}'
        cat_low = f'VIP<{self.vip_threshold}'
        
        df_feat['Cat'] = np.where(
            df_feat['VIP'] >= self.vip_threshold, cat_high, cat_low
        )
        
        palette = {cat_high: 'tab:red', cat_low: 'tab:blue'}
        
        sns.scatterplot(
            data=df_feat, x='Covariance (p1)', y='Correlation (pcorr1)', 
            hue='Cat', palette=palette, edgecolor='k', linewidth=0.5, 
            s=35, ax=ax
        )
        
        # Add quadrant lines
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        self._format_ax(ax, "Covariance", "Pearson's r", "S-Plot")
        self._add_legend(ax, ncol=1, loc='best')

    def plot_vip_bar(self, ax=None, wrap_width=30):
        """Plots a horizontal bar chart of the top N features by VIP score.

        Bars are colored based on their correlation direction (positive=red, 
        negative=blue). Long feature names are automatically wrapped.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes object. 
                If None, creates a new figure.
            wrap_width (int, optional): Max character width before wrapping 
                a label to the next line. Defaults to 20.
        """
        df_feat = self.model.get_features_df(feature_names=self.feature_names)
        top_n = min(self.top_n_vip, len(df_feat))

        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, top_n * 0.22))
        
        df_top = df_feat.head(top_n).sort_values(by='VIP', ascending=True)
        
        # Wrap long feature names using textwrap
        # break_long_words=True ensures continuous chemical names are broken 
        # properly.
        df_top['Feature_Wrapped'] = df_top['Feature'].apply(
            lambda x: textwrap.fill(
                str(x), width=wrap_width, break_long_words=True
            )
        )
        # Color bars based on correlation direction
        df_top['Color'] = df_top['Correlation (pcorr1)'].apply(
            lambda c: 'tab:red' if c > 0 else 'tab:blue'
        )
        bars = ax.barh(
            df_top['Feature_Wrapped'], df_top['VIP'], color=df_top['Color'], 
            edgecolor='k', linewidth=0.5, height=0.8
        )
        # Add precise value labels adjacent to bars
        val_range = df_top['VIP'].max() - df_top['VIP'].min()
        label_jitter = val_range / 50 if len(df_top) > 1 else 0.1
        
        for bar in bars:
            ax.text(
                bar.get_width() + label_jitter, 
                bar.get_y() + bar.get_height() / 2, 
                f'{bar.get_width():.2f}', 
                va='center', ha='left', fontsize=10
            )
                    
        self._format_ax(ax, "VIP", "", "VIP Bar Plot")
        
        # Align y-axis to origin
        ax.spines["left"].set_position(("data", 0))
        ax.tick_params(left=False)
        
        legend_elements = [
            Patch(facecolor='tab:red', edgecolor='k', label='Positive'), 
            Patch(facecolor='tab:blue', edgecolor='k', label='Negative')
        ]
        ax.legend(
            handles=legend_elements, title="Correlation", ncol=1, 
            frameon=True, shadow=True, fontsize=10, borderpad=0.4, 
            facecolor="white", loc="lower right"
        )

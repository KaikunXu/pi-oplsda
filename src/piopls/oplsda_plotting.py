# src/piopls/oplsda_plotting.py
"""Visualization suite for OPLS-DA models.

This module consumes the Pandas DataFrames exported by the OPLSDA model 
to generate publication-ready diagnostic plots, including Model Overview, 
Score Plot, Permutation Test Plot, S-Plot, and VIP Bar Plot.
"""

import textwrap
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.transforms import Affine2D
import seaborn as sns


class OPLSDA_Visualizer:
    """Diagnostic visualizer for fitted OPLS-DA models.

    The visualizer consumes the DataFrame export methods provided by
    :class:`piopls.OPLSDA` and renders a compact diagnostic dashboard covering
    model quality, predictive/orthogonal scores, observation diagnostics,
    permutation results, VIP rankings, and S-plot style feature summaries.

    Args:
        model: Fitted OPLSDA model instance.
        X: Optional feature matrix retained for compatibility with older
            calling code.
        y: Optional class labels used for grouping score plots and legends.
        feature_names: Optional feature names used in exported feature plots.
        sample_names: Optional sample names used in exported sample tables.
        vip_threshold: VIP cutoff used to highlight features in the S-plot.
        top_n_vip: Number of top features displayed in the VIP bar plot.
        custom_palette: Optional explicit class-to-color mapping.

    Attributes:
        model: Fitted OPLSDA model instance.
        feature_names: Feature names passed to feature export methods.
        sample_names: Sample names passed to score export methods.
        class_names: Mapping of encoded integer labels to class labels.
        y_groups: Class assignments used for grouped plots.
        vip_threshold: VIP cutoff for feature highlighting.
        top_n_vip: Maximum number of VIP-ranked features to display.
        palette: Class-to-color mapping used by score and diagnostic plots.
        class_order: Stable class ordering shared by score plot legends.
    """

    PRIMARY_COLOR = "tab:red"
    NEUTRAL_COLOR = "tab:gray"
    LEGEND_DEFAULTS = dict(
        frameon=True,
        shadow=False,
        edgecolor="black",
        fontsize=10,
        title_fontsize=11,
        borderpad=0.4,
        facecolor="white",
        framealpha=1.0,
    )

    def __init__(
        self, 
        model: Any,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        feature_names: Optional[Any] = None,
        sample_names: Optional[Any] = None,
        vip_threshold: float = 1.0,
        top_n_vip: int = 25,
        custom_palette: Optional[dict[Any, str]] = None,
    ) -> None:
        """Initialize the visualizer with model and rendering settings."""
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
        self.LEGEND_KWARGS = self.LEGEND_DEFAULTS.copy()
        
        # Assign color palette dynamically
        if custom_palette is not None:
            self.palette = custom_palette
        else:
            keys = self._ordered_unique(self.y_groups)
            if not keys:
                keys = list(self.class_names.values())
            self.palette = {
                key: self.PRIMARY_COLOR if i == 0 else self.NEUTRAL_COLOR
                for i, key in enumerate(keys)
            }
        self.class_order = [
            key for key in self._ordered_unique(self.y_groups)
            if key in self.palette
        ]
        if not self.class_order:
            self.class_order = list(self.palette.keys())

    @staticmethod
    def _ordered_unique(values: Any) -> list[Any]:
        """Return unique values while preserving their first-seen order."""
        keys = []
        for value in values:
            if value not in keys:
                keys.append(value)
        return keys

    def _draw_confidence_ellipse(
        self,
        x: Any,
        y: Any,
        ax: Any,
        n_std: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Draw a covariance confidence ellipse on an axes object."""
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

    def _format_ax(self, ax: Any, xlabel: str, ylabel: str, title: str) -> None:
        """Applies consistent aesthetic formatting to an axes object."""
        ax.set_xlabel(xlabel, fontsize=11, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='normal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(axis='both', labelsize=11)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    def _add_legend(
        self,
        ax: Any,
        title: str = "",
        loc: str = "best",
        ncol: int = 1,
        handles: Optional[Any] = None,
        labels: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Apply pi-metaboqc-style legend formatting."""
        if handles is None:
            leg = ax.get_legend()
            if leg is not None:
                labels = [text.get_text() for text in leg.get_texts()]
                handles = getattr(
                    leg, "legend_handles", getattr(leg, "legendHandles", [])
                )
                if not handles:
                    handles, labels = ax.get_legend_handles_labels()
                leg.remove()
            else:
                handles, labels = ax.get_legend_handles_labels()
        elif labels is None:
            labels = [handle.get_label() for handle in handles]

        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label and not str(label).startswith("_")
        ]
        if not filtered:
            return None

        handles, labels = zip(*filtered)
        legend_kwargs = self.LEGEND_KWARGS.copy()
        legend_kwargs.update(kwargs)
        legend_kwargs["ncol"] = ncol
        if title:
            legend_kwargs["title"] = title

        return ax.legend(handles, labels, loc=loc, **legend_kwargs)

    def plot_all(
        self,
        perm_results: Optional[dict[str, Any]] = None,
        save_path: Optional[Any] = None,
        wrap_width: int = 20,
        figsize: tuple[float, float] = (8.0, 6.0),
        return_fig: bool = False,
        show_plot: bool = False,
    ) -> Optional[Any]:
        """Generate the combined diagnostic dashboard."""
        import io
        import patchworklib as pw
        import matplotlib.image as mpimg
        
        total_width = figsize[0]
        total_height = figsize[1]
        
        total_parts = 1.0 + 1.0 + 0.8
        w_col1 = total_width * (1.0 / total_parts)
        w_col2 = total_width * (1.0 / total_parts)
        w_col3 = total_width * (0.8 / total_parts)
        
        h_half = total_height / 2.0
        
        ax_ov = pw.Brick(figsize=(w_col1, h_half))
        ax_sc = pw.Brick(figsize=(w_col2, h_half))
        ax_pm = pw.Brick(figsize=(w_col1, h_half))
        ax_ot = pw.Brick(figsize=(w_col2, h_half))
        ax_vp = pw.Brick(figsize=(w_col3, total_height))

        self.plot_model_overview(ax=ax_ov)
        self.plot_score(ax=ax_sc)
        self.plot_outlier(ax=ax_ot)
        self.plot_vip_bar(ax=ax_vp, wrap_width=wrap_width)
        
        if perm_results: 
            self.plot_permutations(perm_results, ax=ax_pm)
        else: 
            ax_pm.axis('off')

        final_fig = ((ax_ov | ax_sc) / (ax_pm | ax_ot)) | ax_vp
        
        if save_path:
            final_fig.savefig(save_path, dpi=300, bbox_inches='tight')

        if show_plot:
            # Render the patchworklib layout to an in-memory buffer
            buf = io.BytesIO()
            final_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Destroy the multiple implicit figures created by patchworklib
            plt.close('all')
            
            # Create a single, clean Figure to display the unified buffer image
            display_fig, display_ax = plt.subplots(figsize=figsize)
            display_ax.imshow(mpimg.imread(buf, format='png'))
            display_ax.axis('off')
            plt.subplots_adjust(
                top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
            )
            
            # Block script execution until the user closes this window
            plt.show(block=True)
            
        if not return_fig:
            return None
            
        # Ensure cleanup occurs if returning the object without showing it
        if not show_plot:
            plt.close('all')
            
        return final_fig


    def plot_model_overview(self, ax: Optional[Any] = None) -> None:
        """Plots a grouped bar chart showing R2Y and Q2 for each component."""
        df_summary = self.model.get_summary_df()
        labels = [
            lbl.split('(')[-1].strip(')') if '(' in lbl else lbl 
            for lbl in df_summary['Component'].tolist()]
        r2y_vals = df_summary['R2Y'].tolist()
        q2_vals  = df_summary['Q2'].tolist()
        
        if ax is None: 
            fig_w = max(4.5, len(labels) * 1.5)
            _, ax = plt.subplots(figsize=(fig_w, 3))
            
        x = np.arange(len(labels))
        width = 0.4

        ax.bar(
            x - width/2, r2y_vals, width, label='R2Y', 
            color=self.NEUTRAL_COLOR, edgecolor='k', linewidth=0.5
        )
        ax.bar(
            x + width/2, q2_vals, width, label='Q2',  
            color=self.PRIMARY_COLOR, edgecolor='k', linewidth=0.5
        )

        def add_labels(vals: list[float], offset: float) -> None:
            """Add numeric labels above or below component bars."""
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
        
        self._format_ax(ax, xlabel="", ylabel="", title="Model Overview")
        self._add_legend(ax, ncol=1, loc='upper right')

    def plot_score(self, ax: Optional[Any] = None) -> None:
        """Plots the sample score scatter plot."""
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_scores = self.model.get_scores_df(
            sample_names=self.sample_names, y_true=self.y_groups
        )
        df_summary = self.model.get_summary_df()
        
        # Compatible modification: access updated keys
        t1 = df_scores['t_pred (p1)']
        p_row = df_summary.loc[
            df_summary['Component'] == 'Predictive (p1)', 'R2X'
        ]
        p_explan = p_row.values[0] * 100
        
        if 't_ortho (o1)' in df_scores.columns:
            o1 = df_scores['t_ortho (o1)']
            o_row = df_summary.loc[
                df_summary['Component'] == 'Orthogonal (o1)', 'R2X'
            ]
            o_explan = o_row.values[0] * 100
        else:
            o1 = np.zeros_like(t1)
            o_explan = 0.0
            
        df_scores['t_ortho_1_plot'] = o1
            
        sns.scatterplot(
            data=df_scores, x='t_pred (p1)', y='t_ortho_1_plot', 
            hue='True_Class', palette=self.palette, edgecolor='k', 
            linewidth=0.5, s=40, ax=ax, hue_order=self.class_order
        )
        
        for cls_name in self.class_order:
            sub = df_scores[df_scores['True_Class'] == cls_name]
            if cls_name in self.palette:
                color = self.palette[cls_name]
                self._draw_confidence_ellipse(
                    sub['t_pred (p1)'], sub['t_ortho_1_plot'], ax, n_std=2.0, 
                    facecolor=color, edgecolor=color, alpha=0.3, linewidth=1
                )

        x_label = f"T Score [1]\n({p_explan:.1f} %)"
        y_label = f"Orthogonal T Score [1]\n({o_explan:.1f} %)"
        self._format_ax(ax, xlabel=x_label, ylabel=y_label, title="X-Score Plot")
        self._add_legend(ax, ncol=1, loc='best')
        
    def plot_permutations(
        self,
        perm_results: dict[str, Any],
        ax: Optional[Any] = None,
    ) -> None:
        """Plots the distribution of R2Y and Q2 from the permutation test."""
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
            
        # Compatible modification: map to the new perm_results keys
        perms_r2y = perm_results.get('perm_R2Y', [])
        perms_q2 = perm_results.get('perm_Q2Y', [])
        real_r2y = perm_results.get('orig_R2Y', 0)
        real_q2 = perm_results.get('orig_Q2Y', 0)
        
        if len(perms_r2y) > 0:
            arr_concat = np.concatenate([perms_r2y, perms_q2])
            bins = np.histogram_bin_edges(arr_concat, bins=30)
            
            ax.hist(
                perms_r2y, bins=bins, color=self.NEUTRAL_COLOR,
                edgecolor='k', linewidth=0.5, label='Perm R2Y'
            )
            ax.hist(
                perms_q2, bins=bins, color=self.PRIMARY_COLOR,
                edgecolor='k', linewidth=0.5, label='Perm Q2'
            )
            
        maxY = ax.get_ylim()[1]
        xmin, xmax = ax.get_xlim()
        needed_xmax = max(real_r2y, real_q2, xmax)
        ax.set_xlim(xmin, needed_xmax + (needed_xmax - xmin) * 0.15)
        
        ax.axvline(
            real_r2y, color=self.NEUTRAL_COLOR, linestyle='--',
            linewidth=1.0, zorder=0
        )
        ax.axvline(
            real_q2, color=self.PRIMARY_COLOR, linestyle='--',
            linewidth=1.0, zorder=0
        )

        arrow = dict(
            arrowstyle="-|>", connectionstyle="arc3, rad=-0.15", 
            color="k", linewidth=1.0
        )
        bbox = dict(boxstyle="round", facecolor="white", edgecolor="k", pad=0.2)
        
        r2y_p = perm_results.get('p_R2Y', 0)
        ax.annotate(
            f"R2Y: {real_r2y:.3f}\nP = {r2y_p:.3f}", 
            xy=(real_r2y, 0), xytext=(real_r2y, maxY * 0.75),
            ha="center", va="center", fontsize=8, bbox=bbox, arrowprops=arrow
        )
        
        q2_p = perm_results.get('p_Q2Y', 0)
        ax.annotate(
            f"Q2: {real_q2:.3f}\nP = {q2_p:.3f}", 
            xy=(real_q2, 0), xytext=(real_q2, maxY * 0.5),
            ha="center", va="center", fontsize=8, bbox=bbox, arrowprops=arrow
        )

        self._format_ax(
            ax, xlabel="Permutations", ylabel="Frequency",
            title="Permutation Test")
        self._add_legend(ax, ncol=1, loc='best')

    def plot_splot(self, ax: Optional[Any] = None) -> None:
        """Plots the S-Plot (Covariance vs Correlation) highlighting features."""
        if ax is None: 
            _, ax = plt.subplots(figsize=(4.5, 3))
        
        df_feat = self.model.get_features_df(feature_names=self.feature_names)
        
        cat_high = f'VIP>={self.vip_threshold}'
        cat_low = f'VIP<{self.vip_threshold}'
        
        df_feat['Cat'] = np.where(
            df_feat['VIP'] >= self.vip_threshold, cat_high, cat_low
        )
        
        palette = {cat_high: self.PRIMARY_COLOR, cat_low: self.NEUTRAL_COLOR}
        
        sns.scatterplot(
            data=df_feat, x='Covariance (p1)', y='Correlation (pcorr1)', 
            hue='Cat', palette=palette, edgecolor='k', linewidth=0.5, 
            s=35, ax=ax
        )
        
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        self._format_ax(
            ax, xlabel="Covariance", ylabel="Pearson's r", title="S-Plot")
        self._add_legend(ax, ncol=1, loc='best')


    def plot_vip_bar(
        self,
        ax: Optional[Any] = None,
        wrap_width: int = 30,
    ) -> None:
        """Plots a bar plot of the top N features by VIP score."""
        df_feat = self.model.get_features_df(feature_names=self.feature_names)
        top_n = min(self.top_n_vip, len(df_feat))

        if ax is None: 
            _, ax = plt.subplots(figsize=(top_n * 0.3, 4.5))
        
        df_top = df_feat.head(top_n).sort_values(by='VIP', ascending=False)
        
        df_top['Feature_Wrapped'] = df_top['Feature'].apply(
            lambda x: textwrap.fill(
                str(x), width=wrap_width, break_long_words=True
            )
        )
        
        df_top['Correlation_Type'] = np.where(
            df_top['Correlation (pcorr1)'] > 0, 'Positive', 'Negative'
        )
        
        sns.barplot(
            data=df_top, 
            y='Feature_Wrapped', 
            x='VIP', 
            hue='Correlation_Type', 
            palette={
                'Positive': self.PRIMARY_COLOR,
                'Negative': self.NEUTRAL_COLOR,
            },
            edgecolor='k', 
            linewidth=0.5, 
            dodge=False,
            ax=ax
        )
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
                    
        self._format_ax(ax, ylabel="", xlabel="VIP", title="VIP Bar Plot")
        
        # ax.set_xticks(
        #     ticks=ax.get_xticks(), labels=ax.get_xticklabels(),
        #     rotation=45, ha='right', rotation_mode='anchor')
        
        ax.spines[["top","right"]].set_visible(False)
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
        legend_elements = [
            Patch(
                facecolor=self.PRIMARY_COLOR, edgecolor='k', label='Positive'
            ),
            Patch(
                facecolor=self.NEUTRAL_COLOR, edgecolor='k', label='Negative'
            )
        ]
        
        self._add_legend(
            ax, handles=legend_elements, title="Correlation", ncol=1, 
            loc='lower right')

    def plot_outlier(self, ax: Optional[Any] = None) -> None:
        """Plot score distance and orthogonal distance diagnostics."""
        from matplotlib.lines import Line2D
        
        if ax is None: 
            _, ax = plt.subplots(figsize=(5.0, 4.0))
            
        df_out: pd.DataFrame = self.model.get_outlier_df(
            sample_names=self.sample_names, y_true=self.y_groups
        )
        
        # Retrieve pre-calculated threshold limits from the model.
        sd_limit: float = getattr(self.model, 'sd_limit_', None)
        od_limit: float = getattr(self.model, 'od_limit_', None)
        
        # 1. Dynamically evaluate if each sample exceeds any threshold limit.
        is_outlier: np.ndarray = np.zeros(len(df_out), dtype=bool)
        if sd_limit is not None:
            is_outlier |= (df_out['Score_Distance'] > sd_limit)
        if od_limit is not None:
            is_outlier |= (df_out['Orthogonal_Distance'] > od_limit)
            
        df_out['Is_Outlier'] = is_outlier
        
        # Isolate data into normal and outlier subsets.
        df_norm: pd.DataFrame = df_out[~df_out['Is_Outlier']]
        df_outl: pd.DataFrame = df_out[df_out['Is_Outlier']]
        
        # 2. Render normal samples: default circle ('o'), size 40.
        if not df_norm.empty:
            sns.scatterplot(
                data=df_norm, x='Score_Distance', y='Orthogonal_Distance', 
                hue='True_Class', palette=self.palette, edgecolor='k', 
                linewidth=0.5, s=40, marker='o', ax=ax, legend=False,
                hue_order=self.class_order
            )
            
        # 3. Render outlier samples: bold cross ('X'), size 70.
        if not df_outl.empty:
            sns.scatterplot(
                data=df_outl, x='Score_Distance', y='Orthogonal_Distance', 
                hue='True_Class', palette=self.palette, edgecolor='k', 
                linewidth=0.5, s=70, marker='X', ax=ax, legend=False,
                hue_order=self.class_order
            )
            
        # Draw black threshold cutoff lines.
        if sd_limit is not None:
            ax.axvline(
                x=sd_limit, color='k', linestyle='--', 
                linewidth=1.2, zorder=0
            )
        if od_limit is not None:
            ax.axhline(
                y=od_limit, color='k', linestyle=':', 
                linewidth=1.2, zorder=0
            )
            
        self._format_ax(
            ax, "Score Distance (SD)", "Orthogonal Distance (OD)", 
            "Observation Diagnostics"
        )
        
        # 4. Manually assemble the right-side legend.
        legend_elements: list = []
        
        # Add color-coded circular markers for true classes.
        for cls_name in self.class_order:
            if cls_name in self.palette:
                color: str = self.palette[cls_name]
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=cls_name,
                           markerfacecolor=color, markeredgecolor='k', 
                           markersize=7)
                )
                
        # Append a shape indicator for the outlier 'X' marker.
        legend_elements.append(
            Line2D([0], [0], marker='X', color='w', label='Outliers',
                   markerfacecolor='white', markeredgecolor='k',
                   markersize=8)
        )
        
        self._add_legend(
            ax, handles=legend_elements, title="", ncol=1, 
            loc="upper right"
        )

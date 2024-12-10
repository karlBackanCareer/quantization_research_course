import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


class ImageQualityAnalyzer:
    def __init__(self, base_dir='testResults'):
        self.base_dir = Path(base_dir)
        self.participant_level_data = []
        self.rating_details = []
        self.analysis_results = {}

        # Custom color palette for improved visualization
        self.colors = {
            'original': '#2ecc71',
            'scale_0.85': '#3498db',
            'scale_0.66': '#e74c3c',
            'scale_0.50': '#f1c40f',
            'scale_0.33': '#9b59b6',
            'method1': '#2980b9',
            'method2': '#c0392b'
        }

    def create_visualizations(self, output_dir='analysis_output'):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Simplified Rating Trends Over Time
        plt.figure(figsize=(14, 8))
        plot_df = self.rating_df.copy()
        plot_df.loc[plot_df['is_original'], 'scale'] = 1.0
        window = 15  # Increased window for smoother trends

        # Plot with reduced points to minimize clutter
        step = 3
        for scale in sorted(plot_df['scale'].unique()):
            data = plot_df[plot_df['scale'] == scale]
            ratings = data['rating'].rolling(window=window, min_periods=1).mean()
            x_vals = data.index[::step]
            y_vals = ratings.values[::step]
            scale_label = 'Original' if scale == 1.0 else f'Scale {scale:.2f}'
            color = self.colors['original'] if scale == 1.0 else self.colors[f'scale_{scale:.2f}']
            plt.plot(x_vals, y_vals, label=scale_label, color=color, alpha=0.8)

        plt.title('Rating Trends Over Time (Smoothed)', pad=20)
        plt.xlabel('Rating Sequence')
        plt.ylabel('Average Rating')
        plt.legend(title='Scale Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Enhanced Method Comparison
        plt.figure(figsize=(12, 8))
        plot_df_method = plot_df.copy()
        plot_df_method['method_scale'] = plot_df_method.apply(
            lambda x: f"{x['method']} ({x['scale']:.2f})" if not x['is_original']
            else 'Original', axis=1)

        # Bar plot with error bars and individual points
        sns.barplot(data=plot_df_method,
                    x='scale', y='rating',
                    hue='method',
                    palette=[self.colors['method1'], self.colors['method2']],
                    ci=95,
                    capsize=0.1,
                    alpha=0.8)

        sns.stripplot(data=plot_df_method,
                      x='scale', y='rating',
                      hue='method',
                      dodge=True,
                      alpha=0.2,
                      size=4)

        plt.title('Average Quality Rating by Scale and Method', pad=20)
        plt.xlabel('Scale Factor')
        plt.ylabel('Quality Rating')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300)
        plt.close()

        # 3. Simplified Distribution Plot
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=self.rating_df,
                       x='image', y='rating',
                       hue='is_original',
                       split=True,
                       inner=None,
                       palette=[self.colors['method1'], self.colors['original']])

        plt.title('Rating Distribution by Image Type', pad=20)
        plt.xlabel('Image Type')
        plt.ylabel('Quality Rating')
        plt.xticks(rotation=45)
        plt.legend(title='Original Image')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300)
        plt.close()

        # 4. Enhanced Quality Preservation Trend
        plt.figure(figsize=(12, 8))
        trend_df = plot_df.groupby('scale').agg({
            'rating': ['mean', 'std', 'count']
        }).reset_index()
        trend_df.columns = ['scale', 'mean_rating', 'std_rating', 'count']

        orig_mean = trend_df[trend_df['scale'] == 1.0]['mean_rating'].values[0]
        trend_df['quality_preserved'] = (trend_df['mean_rating'] / orig_mean) * 100
        trend_df = trend_df.sort_values('scale')

        plt.fill_between(trend_df['scale'],
                         trend_df['quality_preserved'] - trend_df['std_rating'],
                         trend_df['quality_preserved'] + trend_df['std_rating'],
                         alpha=0.2, color=self.colors['method1'])

        plt.plot(trend_df['scale'],
                 trend_df['quality_preserved'],
                 color=self.colors['method1'],
                 linewidth=2.5,
                 marker='o',
                 markersize=8)

        plt.title('Perceived Quality Preservation by Scale Factor', pad=20)
        plt.xlabel('Scale Factor')
        plt.ylabel('Quality Preserved (%)')
        plt.grid(True, alpha=0.3)
        x_ticks = trend_df['scale'].values
        x_labels = ['Original' if x == 1.0 else f'{x:.2f}' for x in x_ticks]
        plt.xticks(x_ticks, x_labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_preservation.png'), dpi=300)
        plt.close()

    def perform_statistical_analysis(self):
        """Enhanced statistical analysis with additional metrics"""
        if not hasattr(self, 'rating_df') or self.rating_df.empty:
            return None

        # Basic scale effects
        scale_effects = self._analyze_scale_effects()

        # Add pairwise comparisons
        scale_data = self.rating_df[~self.rating_df['is_original']]
        scales = sorted(scale_data['scale'].unique())
        scale_comparisons = {}

        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):
                scale1, scale2 = scales[i], scales[j]
                group1 = scale_data[scale_data['scale'] == scale1]['rating']
                group2 = scale_data[scale_data['scale'] == scale2]['rating']

                t_stat, p_val = stats.ttest_ind(group1, group2)
                effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt(
                    (np.var(group1) + np.var(group2)) / 2)

                scale_comparisons[(scale1, scale2)] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'effect_size': effect_size
                }

        # Add inter-rater reliability
        ratings_wide = self.rating_df.pivot_table(
            index=['image', 'scale', 'method'],
            columns='participant_id',
            values='rating'
        )
        icc = self._calculate_icc(ratings_wide)

        self.analysis_results = {
            'scale_effects': scale_effects,
            'scale_comparisons': scale_comparisons,
            'icc': icc,
            'method_effects': self._analyze_method_effects(),
            'image_type_effects': self._analyze_image_type_effects()
        }

        return self.analysis_results

    def _calculate_icc(self, ratings):
        """Calculate Intraclass Correlation Coefficient"""
        ratings = ratings.dropna()
        if ratings.empty:
            return None

        n = len(ratings.columns)  # number of raters
        k = len(ratings)  # number of subjects

        # Calculate mean squares
        between_subject_ss = ((ratings.mean(axis=1) - ratings.values.mean()) ** 2).sum() * n
        within_subject_ss = ((ratings - ratings.mean(axis=1).values.reshape(-1, 1)) ** 2).sum().sum()

        between_subject_ms = between_subject_ss / (k - 1)
        within_subject_ms = within_subject_ss / (k * (n - 1))

        # Calculate ICC(2,1)
        icc = (between_subject_ms - within_subject_ms) / (between_subject_ms + (n - 1) * within_subject_ms)

        return icc

    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            print("No analysis results available.")
            return

        print("\n=== Image Quality Analysis Summary ===\n")

        # Scale effects
        se = self.analysis_results['scale_effects']
        print("SCALE EFFECTS:")
        print(f"ANOVA: F={se['anova']['f_stat']:.3f}, p={se['anova']['p_value']:.3f}")
        print(f"Effect size (η²): {se['anova']['effect_size']:.3f}\n")

        # Scale comparisons
        print("PAIRWISE SCALE COMPARISONS:")
        for (scale1, scale2), stats in self.analysis_results['scale_comparisons'].items():
            print(f"{scale1} vs {scale2}:")
            print(f"  t={stats['t_stat']:.3f}, p={stats['p_value']:.3f}")
            print(f"  Effect size (Cohen's d): {stats['effect_size']:.3f}\n")

        # Inter-rater reliability
        if self.analysis_results['icc'] is not None:
            print(f"INTER-RATER RELIABILITY:")
            print(f"ICC(2,1): {self.analysis_results['icc']:.3f}\n")

        # Method effects
        if self.analysis_results['method_effects']:
            me = self.analysis_results['method_effects']
            print("METHOD EFFECTS:")
            print(f"ANOVA: F={me['f_stat']:.3f}, p={me['p_value']:.3f}\n")

        # Quality preservation by scale
        print("QUALITY PRESERVATION BY SCALE:")
        metrics = self.analysis_results['scale_effects']['scale_metrics']
        orig_mean = next(m['mean_rating'] for m in metrics if m['scale'] == 1.0)
        for metric in metrics:
            quality_preserved = (metric['mean_rating'] / orig_mean) * 100
            ci = 1.96 * metric['std_rating'] / np.sqrt(metric['count'])
            scale_label = 'Original' if metric['scale'] == 1.0 else f"Scale {metric['scale']:.2f}"
            print(f"{scale_label}:")
            print(f"  Mean Rating: {metric['mean_rating']:.2f} ± {metric['std_rating']:.2f}")
            print(f"  Quality Preserved: {quality_preserved:.1f}%")
            print(f"  95% CI: [{quality_preserved - ci:.1f}%, {quality_preserved + ci:.1f}%]\n")

    def load_and_process_data(self):
        """Load and process all test data from directories."""
        for group_dir in ['family', 'friendGroup', 'projectTeam']:
            directory = self.base_dir / group_dir
            if directory.exists():
                print(f"Loading {group_dir}...")
                self._process_group_directory(directory, group_dir)

        if self.participant_level_data:
            self.participant_df = pd.DataFrame(self.participant_level_data)
            self.rating_df = pd.DataFrame(self.rating_details)

            # Add normalized ratings per participant
            self.rating_df['rating_normalized'] = self.rating_df.groupby('participant_id')['rating'].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 else x
            )

            # Add additional computed columns
            self._add_computed_columns()
            return True

        print("No data was loaded. Check directory structure and file paths.")
        return False

    def _process_group_directory(self, directory, group_name):
        """Process all JSON files in a group directory."""
        json_files = list(directory.glob('*.json'))
        if not json_files:
            print(f"No JSON files found in {directory}")
            return

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._process_participant_data(data, group_name)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    def _process_participant_data(self, data, group_name):
        """Process individual participant data."""
        try:
            participant_id = data['id']
            results = data.get('results', [])

            # Basic participant summary
            participant_summary = {
                'participant_id': participant_id,
                'group': group_name,
                'total_time': (pd.to_datetime(data.get('endTime')) -
                               pd.to_datetime(data.get('startTime'))).total_seconds(),
                'num_ratings': len(results)
            }

            # Process individual ratings
            for result in results:
                if not isinstance(result, dict):
                    continue

                rating_detail = {
                    'participant_id': participant_id,
                    'group': group_name,
                    'image': result.get('image'),
                    'rating': result.get('rating'),
                    'response_time': result.get('responseTimeSeconds'),
                    'is_original': result.get('isOriginal', False),
                    'scale': result.get('scale'),
                    'method': result.get('method'),
                    'sequence_num': len(self.rating_details)
                }

                if rating_detail['rating'] is None or rating_detail['image'] is None:
                    continue

                self.rating_details.append(rating_detail)

            # Calculate participant-level metrics
            orig_ratings = [r['rating'] for r in results if r.get('isOriginal', False)]
            participant_summary['avg_original_rating'] = np.mean(orig_ratings) if orig_ratings else np.nan

            # Add scale-specific metrics
            for scale in [0.85, 0.66, 0.5, 0.33]:
                scale_ratings = [r['rating'] for r in results
                                 if r.get('scale') == scale and not r.get('isOriginal', False)]
                if scale_ratings:
                    participant_summary[f'avg_scale_{scale}'] = np.mean(scale_ratings)
                    participant_summary[f'std_scale_{scale}'] = np.std(scale_ratings)

            self.participant_level_data.append(participant_summary)

        except Exception as e:
            print(f"Error processing participant data: {str(e)}")

    def _add_computed_columns(self):
        """Add computed columns for analysis."""
        # Calculate mean ratings for original images
        orig_mean = self.rating_df[self.rating_df['is_original']]['rating'].mean()

        # Process scale data including originals
        scale_data = self.rating_df.copy()
        scale_data.loc[scale_data['is_original'], 'scale'] = 1.0

        scale_stats = scale_data.groupby('scale').agg({
            'rating': ['mean', 'std', 'count']
        }).reset_index()
        scale_stats.columns = ['scale', 'mean_rating', 'std_rating', 'count']

        # Calculate quality preservation relative to original
        orig_mean = scale_stats[scale_stats['scale'] == 1.0]['mean_rating'].values[0]
        scale_stats['quality_preserved'] = (scale_stats['mean_rating'] / orig_mean) * 100

        # Store for later use
        self.scale_analysis = scale_stats

        # Add sequence-based features
        self.rating_df['prev_rating'] = self.rating_df.groupby('participant_id')['rating'].shift(1)
        self.rating_df['rating_diff'] = self.rating_df['rating'] - self.rating_df['prev_rating']

    def _analyze_scale_effects(self):
        """Analyze the effect of scaling on image quality."""
        scale_data = self.rating_df[~self.rating_df['is_original']]
        scales = sorted(scale_data['scale'].unique())

        # ANOVA across scales
        scale_groups = [scale_data[scale_data['scale'] == scale]['rating'].values
                        for scale in scales]
        f_stat, p_val = f_oneway(*scale_groups)

        # Calculate effect size (eta-squared)
        total_ss = np.sum((scale_data['rating'] - scale_data['rating'].mean()) ** 2)
        between_ss = sum(len(group) * (np.mean(group) - scale_data['rating'].mean()) ** 2
                         for group in scale_groups)
        eta_squared = between_ss / total_ss if total_ss != 0 else np.nan

        return {
            'anova': {
                'f_stat': f_stat,
                'p_value': p_val,
                'effect_size': eta_squared
            },
            'scale_metrics': self.scale_analysis.to_dict('records')
        }

    def _analyze_method_effects(self):
        """Analyze the effect of interpolation method."""
        if 'method' not in self.rating_df.columns:
            return None

        method_data = self.rating_df[~self.rating_df['is_original']]
        methods = method_data['method'].unique()

        if len(methods) < 2:
            return None

        method_stats = {}
        for method in methods:
            method_ratings = method_data[method_data['method'] == method]['rating']
            method_stats[method] = {
                'mean': method_ratings.mean(),
                'std': method_ratings.std(),
                'n': len(method_ratings)
            }

        # Perform ANOVA
        method_groups = [method_data[method_data['method'] == method]['rating'].values
                         for method in methods]
        f_stat, p_val = f_oneway(*method_groups)

        return {
            'method_stats': method_stats,
            'f_stat': f_stat,
            'p_value': p_val
        }

    def _analyze_image_type_effects(self):
        """Analyze effects across different image types."""
        image_data = self.rating_df[~self.rating_df['is_original']]
        image_stats = {}

        for image in image_data['image'].unique():
            image_ratings = image_data[image_data['image'] == image]['rating']
            image_stats[image] = {
                'mean': image_ratings.mean(),
                'std': image_ratings.std(),
                'n': len(image_ratings)
            }

        # Perform ANOVA
        image_groups = [group['rating'].values for name, group
                        in image_data.groupby('image')]

        if len(image_groups) < 2:
            return None

        f_stat, p_val = f_oneway(*image_groups)

        # Add post-hoc analysis
        tukey = pairwise_tukeyhsd(image_data['rating'], image_data['image'])

        return {
            'image_stats': image_stats,
            'f_stat': f_stat,
            'p_value': p_val,
            'tukey_results': tukey
        }


def main():
    analyzer = ImageQualityAnalyzer()
    if analyzer.load_and_process_data():
        analyzer.perform_statistical_analysis()
        analyzer.create_visualizations()
        analyzer.print_analysis_summary()


if __name__ == "__main__":
    main()
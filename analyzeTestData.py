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

        # Enhanced contrast color palette
        self.colors = {
            'original': '#1a73e8',  # Bright blue
            'scale_0.85': '#34a853',  # Vivid green
            'scale_0.66': '#ea4335',  # Bright red
            'scale_0.50': '#fbbc04',  # Bright yellow
            'scale_0.33': '#673ab7',  # Deep purple
            'method1': '#2196f3',  # Light blue
            'method2': '#f44336'  # Light red
        }

        # Enhanced default plot styling
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 14,  # Increased base font size
            'axes.labelsize': 16,  # Larger axis labels
            'axes.titlesize': 18,  # Larger titles
            'xtick.labelsize': 14,  # Larger tick labels
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'figure.autolayout': True
        })

    def create_visualizations(self, output_dir='analysis_output'):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Enhanced Rating Trends
        plt.figure(figsize=(16, 10))
        plot_df = self.rating_df.copy()
        plot_df.loc[plot_df['is_original'], 'scale'] = 1.0
        window = 20

        for scale in sorted(plot_df['scale'].unique()):
            data = plot_df[plot_df['scale'] == scale]
            ratings = data['rating'].rolling(window=window, min_periods=1).mean()
            scale_label = 'Original' if scale == 1.0 else f'Scale {scale:.2f}'
            color = self.colors['original'] if scale == 1.0 else self.colors[f'scale_{scale:.2f}']

            plt.plot(data.index, ratings,
                     label=scale_label,
                     color=color,
                     alpha=0.9,
                     linewidth=3)

        plt.title('Image Quality Rating Trends', pad=20, fontweight='bold')
        plt.xlabel('Rating Sequence', labelpad=15)
        plt.ylabel('Average Rating', labelpad=15)

        plt.legend(title='Scale Factor',
                   title_fontsize=14,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   frameon=True,
                   edgecolor='black',
                   fancybox=False,
                   shadow=True)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_dir, 'rating_trends.png'), bbox_inches='tight')
        plt.close()

        # 2. Enhanced Method Comparison
        plt.figure(figsize=(12, 8))
        plot_df_method = plot_df.copy()
        plot_df_method = plot_df_method[plot_df_method['method'].notna()]

        ax = sns.barplot(data=plot_df_method,
                         x='scale',
                         y='rating',
                         hue='method',
                         palette=[self.colors['method1'], self.colors['method2']],
                         ci=95,
                         capsize=0.1,
                         errwidth=2,
                         alpha=0.8)

        plt.ylim(0, 5)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=12, fontweight='bold')

        plt.title('Quality Ratings by Scale and Method', pad=20, fontweight='bold')
        plt.xlabel('Scale Factor', labelpad=15)
        plt.ylabel('Quality Rating', labelpad=15)

        plt.legend(title='Method',
                   title_fontsize=14,
                   frameon=True,
                   edgecolor='black',
                   fancybox=False,
                   shadow=True,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), bbox_inches='tight')
        plt.close()

        # 3. Enhanced Distribution Plot
        plt.figure(figsize=(16, 10))
        sns.violinplot(data=self.rating_df,
                       x='image',
                       y='rating',
                       hue='is_original',
                       split=True,
                       inner=None,
                       palette=['#ea4335', '#34a853'])  # Red for processed, green for original

        plt.title('Rating Distribution by Image Type', pad=20, fontweight='bold')
        plt.xlabel('Image Type', labelpad=15)
        plt.ylabel('Quality Rating', labelpad=15)
        plt.xticks(rotation=45, ha='right')

        plt.legend(title='Original Image',
                   title_fontsize=14,
                   frameon=True,
                   edgecolor='black',
                   fancybox=False,
                   shadow=True)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), bbox_inches='tight')
        plt.close()

        # 4. Mean Quality Ratings Trend
        plt.figure(figsize=(14, 10))
        trend_df = plot_df.groupby('scale').agg({
            'rating': ['mean', 'std', 'count']
        }).reset_index()
        trend_df.columns = ['scale', 'mean_rating', 'std_rating', 'count']
        trend_df = trend_df.sort_values('scale')

        plt.fill_between(trend_df['scale'],
                         trend_df['mean_rating'] - trend_df['std_rating'],
                         trend_df['mean_rating'] + trend_df['std_rating'],
                         alpha=0.2,
                         color=self.colors['method1'],
                         label='Standard Deviation')

        plt.plot(trend_df['scale'],
                 trend_df['mean_rating'],
                 color=self.colors['method1'],
                 linewidth=3,
                 marker='o',
                 markersize=10,
                 label='Mean Quality Rating')

        for x, y in zip(trend_df['scale'], trend_df['mean_rating']):
            plt.annotate(f'{y:.2f}',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=12,
                         fontweight='bold')

        plt.title('Mean Quality Rating by Scale Factor',
                  pad=20,
                  fontweight='bold')
        plt.xlabel('Scale Factor', labelpad=15)
        plt.ylabel('Mean Quality Rating', labelpad=15)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True,
                   edgecolor='black',
                   fancybox=False,
                   shadow=True,
                   loc='best')

        plt.ylim(0, 5)  # Set y-axis to match rating scale

        x_ticks = trend_df['scale'].values
        x_labels = ['Original' if x == 1.0 else f'{x:.2f}' for x in x_ticks]
        plt.xticks(x_ticks, x_labels)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True,
                   edgecolor='black',
                   fancybox=False,
                   shadow=True,
                   loc='best')

        x_ticks = trend_df['scale'].values
        x_labels = ['Original' if x == 1.0 else f'{x:.2f}' for x in x_ticks]
        plt.xticks(x_ticks, x_labels)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_preservation.png'), bbox_inches='tight')
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

    def _analyze_image_characteristics(self):
        """Analyzes specific characteristics for each image type"""
        if not hasattr(self, 'rating_df') or self.rating_df.empty:
            return None

        image_characteristics = {}

        # Analyze each image type separately
        for image in self.rating_df['image'].unique():
            image_data = self.rating_df[self.rating_df['image'] == image]

            # Calculate basic statistics
            overall_stats = {
                'mean': image_data['rating'].mean(),
                'std': image_data['rating'].std(),
                'n': len(image_data)
            }

            # Calculate method sensitivity
            method_groups = [group['rating'].values for name, group
                             in image_data[~image_data['is_original']].groupby('method')]
            if len(method_groups) >= 2:
                f_stat, p_val = f_oneway(*method_groups)
                method_sensitivity = {
                    'f_stat': f_stat,
                    'p_value': p_val
                }
            else:
                method_sensitivity = None

            # Calculate scale degradation
            scale_means = image_data[~image_data['is_original']].groupby('scale')['rating'].mean()
            scale_degradation = (scale_means.max() - scale_means.min()) / scale_means.max()

            # Calculate scale consistency
            scale_stds = image_data[~image_data['is_original']].groupby('scale')['rating'].std()
            scale_consistency = scale_stds.mean()

            image_characteristics[image] = {
                'stats': overall_stats,
                'method_sensitivity': method_sensitivity,
                'scale_degradation': scale_degradation,
                'scale_consistency': scale_consistency
            }

        return image_characteristics

    def _analyze_image_type_effects(self):
        """Enhanced analysis of effects across different image types."""
        image_data = self.rating_df[~self.rating_df['is_original']]
        image_stats = {}

        # Basic statistics per image
        for image in image_data['image'].unique():
            image_ratings = image_data[image_data['image'] == image]['rating']
            image_stats[image] = {
                'mean': image_ratings.mean(),
                'std': image_ratings.std(),
                'n': len(image_ratings)
            }

        # Get detailed characteristics
        characteristics = self._analyze_image_characteristics()

        # Categorize images based on characteristics
        categorized_results = {
            'high_detail_images': {},  # baboon, etc.
            'structural_images': {},  # bridge, etc.
            'smooth_images': {},  # pepper, etc.
            'pattern_images': {}  # zebra, etc.
        }

        for image, data in characteristics.items():
            # High detail category (like baboon)
            if data['stats']['std'] > 0.7 and data['scale_degradation'] > 0.5:
                categorized_results['high_detail_images'][image] = data
            # Structural category (like bridge)
            elif data['scale_consistency'] < 0.5 and data['stats']['std'] < 0.5:
                categorized_results['structural_images'][image] = data
            # Add other categorizations as needed

        # Perform ANOVA
        image_groups = [group['rating'].values for name, group
                        in image_data.groupby('image')]
        f_stat, p_val = f_oneway(*image_groups)

        # Add post-hoc analysis
        tukey = pairwise_tukeyhsd(image_data['rating'], image_data['image'])

        return {
            'image_stats': image_stats,
            'f_stat': f_stat,
            'p_value': p_val,
            'tukey_results': tukey,
            'detailed_characteristics': characteristics,
            'categorized_results': categorized_results
        }

    def print_image_analysis_summary(self):
        """Print detailed summary of image-specific analysis"""
        if not hasattr(self, 'analysis_results'):
            print("No analysis results available.")
            return

        image_effects = self.analysis_results.get('image_type_effects')
        if not image_effects:
            return

        print("\nIMAGE-SPECIFIC ANALYSIS:")

        categories = image_effects['categorized_results']

        # Print high detail images (like baboon)
        if categories['high_detail_images']:
            print("\nComplex Texture Patterns:")
            for image, data in categories['high_detail_images'].items():
                print(f"\n{image.capitalize()}:")
                print(f"  Rating variance: σ = {data['stats']['std']:.2f}")
                print(f"  Scale degradation: {data['scale_degradation'] * 100:.1f}%")
                if data['method_sensitivity']:
                    print(f"  Method sensitivity: p = {data['method_sensitivity']['p_value']:.3f}")

        # Print structural images (like bridge)
        if categories['structural_images']:
            print("\nStructural Content:")
            for image, data in categories['structural_images'].items():
                print(f"\n{image.capitalize()}:")
                print(f"  Rating consistency: σ = {data['stats']['std']:.2f}")
                print(f"  Scale consistency: {data['scale_consistency']:.3f}")
                if data['method_sensitivity']:
                    print(f"  Method independence: p = {data['method_sensitivity']['p_value']:.3f}")
    def analyze_group_differences(self):
        """Analyze differences and correlations between participant groups"""
        if not hasattr(self, 'rating_df') or self.rating_df.empty:
            return None

        results = {
            'correlations': {},
            'group_comparisons': {},
            'rating_patterns': {},
            'consistency': {}
        }

        # 1. Within-group rating correlations
        group_correlations = {}
        for group in self.rating_df['group'].unique():
            group_data = self.rating_df[self.rating_df['group'] == group].pivot_table(
                index=['image', 'scale', 'method'],
                columns='participant_id',
                values='rating'
            )
            # Calculate correlation matrix and average correlation
            corr_matrix = group_data.corr()
            # Get upper triangle values excluding diagonal
            upper_triangle = np.triu(corr_matrix, k=1)
            avg_corr = np.nanmean(upper_triangle)
            group_correlations[group] = {
                'avg_correlation': avg_corr,
                'correlation_matrix': corr_matrix,
                'std_correlation': np.nanstd(upper_triangle)
            }
        results['correlations']['within_group'] = group_correlations

        # 2. Between-group comparisons
        # ANOVA between groups
        group_ratings = [group_data['rating'].values for name, group_data
                         in self.rating_df.groupby('group')]
        f_stat, p_val = f_oneway(*group_ratings)
        results['group_comparisons']['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_val
        }

        # Pairwise t-tests between groups
        groups = sorted(self.rating_df['group'].unique())
        t_test_results = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                ratings1 = self.rating_df[self.rating_df['group'] == group1]['rating']
                ratings2 = self.rating_df[self.rating_df['group'] == group2]['rating']

                t_stat, p_val = stats.ttest_ind(ratings1, ratings2)
                # Calculate Cohen's d effect size
                effect_size = (np.mean(ratings1) - np.mean(ratings2)) / np.sqrt(
                    ((len(ratings1) - 1) * np.var(ratings1) +
                     (len(ratings2) - 1) * np.var(ratings2)) /
                    (len(ratings1) + len(ratings2) - 2)
                )

                t_test_results[f'{group1}_vs_{group2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'effect_size': effect_size
                }
        results['group_comparisons']['t_tests'] = t_test_results

        # 3. Tukey's HSD test
        tukey = pairwise_tukeyhsd(self.rating_df['rating'], self.rating_df['group'])
        results['group_comparisons']['tukey_hsd'] = tukey

        # 4. Rating patterns analysis
        for group in self.rating_df['group'].unique():
            group_data = self.rating_df[self.rating_df['group'] == group]

            # Calculate basic statistics
            stats_summary = {
                'mean_rating': group_data['rating'].mean(),
                'std_rating': group_data['rating'].std(),
                'median_rating': group_data['rating'].median(),
                'rating_range': group_data['rating'].max() - group_data['rating'].min(),
                'n_ratings': len(group_data)
            }

            # Scale-specific statistics
            scale_stats = {}
            for scale in sorted(group_data['scale'].unique()):
                if pd.isna(scale):
                    continue
                scale_data = group_data[group_data['scale'] == scale]
                scale_stats[str(scale)] = {
                    'mean': scale_data['rating'].mean(),
                    'std': scale_data['rating'].std(),
                    'n': len(scale_data)
                }

            results['rating_patterns'][group] = {
                'summary': stats_summary,
                'scale_statistics': scale_stats
            }

        # 5. Consistency analysis
        for group in self.rating_df['group'].unique():
            group_data = self.rating_df[self.rating_df['group'] == group]

            # Calculate ICC
            group_wide = group_data.pivot_table(
                index=['image', 'scale', 'method'],
                columns='participant_id',
                values='rating'
            )
            icc = self._calculate_icc(group_wide)

            # Calculate coefficient of variation for ratings
            cv = group_data.groupby('participant_id')['rating'].std() / \
                 group_data.groupby('participant_id')['rating'].mean()

            # Calculate agreement on original images
            orig_data = group_data[group_data['is_original']].pivot_table(
                index='image',
                columns='participant_id',
                values='rating'
            )
            orig_agreement = np.mean([
                len(row[abs(row - row.mean()) <= 1]) / len(row)
                for _, row in orig_data.iterrows()
            ])

            results['consistency'][group] = {
                'icc': icc,
                'cv_mean': cv.mean(),
                'cv_std': cv.std(),
                'original_agreement': orig_agreement
            }

        self._create_group_comparison_visualizations()
        return results

    def _create_group_comparison_visualizations(self, output_dir='analysis_output'):
        """Create comprehensive visualizations for group comparisons"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Rating distributions by group
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=self.rating_df, x='group', y='rating')
        plt.title('Rating Distributions by Group')
        plt.xlabel('Group')
        plt.ylabel('Rating')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_rating_distributions.png'), dpi=300)
        plt.close()

        # 2. Scale-specific comparisons
        plt.figure(figsize=(15, 8))
        for group in sorted(self.rating_df['group'].unique()):
            group_data = self.rating_df[self.rating_df['group'] == group].copy()
            group_data.loc[group_data['is_original'], 'scale'] = 1.0
            means = group_data.groupby('scale')['rating'].mean()
            std = group_data.groupby('scale')['rating'].std()

            plt.errorbar(means.index, means.values,
                         yerr=std.values,
                         label=group,
                         marker='o',
                         capsize=5,
                         linewidth=2,
                         markersize=8)

        plt.title('Quality Ratings by Scale and Group')
        plt.xlabel('Scale Factor')
        plt.ylabel('Average Rating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_scale_comparison.png'), dpi=300)
        plt.close()

        # 3. Correlation heatmaps
        for group in sorted(self.rating_df['group'].unique()):
            group_data = self.rating_df[self.rating_df['group'] == group].pivot_table(
                index=['image', 'scale', 'method'],
                columns='participant_id',
                values='rating'
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(group_data.corr(),
                        annot=True,
                        cmap='RdYlBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1)
            plt.title(f'Rating Correlations - {group}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{group}.png'), dpi=300)
            plt.close()

        # 4. Method comparison across groups
        plt.figure(figsize=(12, 8))
        method_data = self.rating_df[~self.rating_df['is_original']]
        sns.boxplot(data=method_data, x='group', y='rating', hue='method')
        plt.title('Rating Distribution by Group and Method')
        plt.xlabel('Group')
        plt.ylabel('Rating')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'group_method_comparison.png'), dpi=300)
        plt.close()

    def print_group_analysis_summary(self):
        """Print comprehensive summary of group analysis results"""
        results = self.analyze_group_differences()
        if not results:
            print("No analysis results available.")
            return

        print("\n=== Group Analysis Summary ===\n")

        # 1. Group Correlations
        print("WITHIN-GROUP CORRELATIONS:")
        for group, corr_data in results['correlations']['within_group'].items():
            print(f"\n{group}:")
            print(f"  Average correlation: {corr_data['avg_correlation']:.3f}")
            print(f"  Correlation std dev: {corr_data['std_correlation']:.3f}")

        # 2. Group Comparisons
        print("\nGROUP COMPARISONS:")
        print("\nANOVA Results:")
        anova = results['group_comparisons']['anova']
        print(f"  F-statistic: {anova['f_statistic']:.3f}")
        print(f"  p-value: {anova['p_value']:.3f}")

        print("\nPairwise T-tests:")
        for comparison, stats in results['group_comparisons']['t_tests'].items():
            print(f"\n{comparison}:")
            print(f"  t-statistic: {stats['t_statistic']:.3f}")
            print(f"  p-value: {stats['p_value']:.3f}")
            print(f"  Effect size (Cohen's d): {stats['effect_size']:.3f}")

        # 3. Consistency Metrics
        print("\nGROUP CONSISTENCY METRICS:")
        for group, metrics in results['consistency'].items():
            print(f"\n{group}:")
            print(f"  ICC: {metrics['icc']:.3f}")
            print(f"  Coefficient of Variation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
            print(f"  Original Image Agreement: {metrics['original_agreement']:.3f}")

        # 4. Rating Patterns
        print("\nRATING PATTERNS BY GROUP:")
        for group, patterns in results['rating_patterns'].items():
            print(f"\n{group}:")
            summary = patterns['summary']
            print(f"  Mean Rating: {summary['mean_rating']:.2f} ± {summary['std_rating']:.2f}")
            print(f"  Median Rating: {summary['median_rating']:.2f}")
            print(f"  Rating Range: {summary['rating_range']:.2f}")
            print(f"  Number of Ratings: {summary['n_ratings']}")


def main():
    analyzer = ImageQualityAnalyzer()
    if analyzer.load_and_process_data():
        analyzer.perform_statistical_analysis()
        analyzer.create_visualizations()
        analyzer.print_analysis_summary()
        analyzer.analyze_group_differences()
        analyzer.print_group_analysis_summary()


if __name__ == "__main__":
    main()
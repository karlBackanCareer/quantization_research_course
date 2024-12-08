import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # From transcript: "three steps or four steps", "four or five different scales"
        self.scale_factors = [0.85, 0.66, 0.5, 0.33]

        # From transcript: "landing source four and bicubic"
        self.upscale_methods = [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

        # From transcript: "two images on smooth areas, snow sky", "two images with kind of a lot going on"
        self.categories = {
            'smooth': ['snow', 'sky'],
            'busy': ['forest', 'baboon', 'leaves'],
            'mixed': ['bridge', 'building']
        }

    def process_dataset(self):
        results = []
        image_files = list(self.input_dir.glob('*.jpg')) + list(self.input_dir.glob('*.png'))

        for img_path in image_files:

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # From transcript: "keep them at same square aspect ratio"
            h, w = img.shape[:2]
            min_dim = min(h, w)
            img = img[:min_dim, :min_dim]

            original_path = self.output_dir / f"{img_path.stem}_original.jpg"
            cv2.imwrite(str(original_path), img)

            for scale in self.scale_factors:
                down_size = (int(min_dim * scale), int(min_dim * scale))
                downscaled = cv2.resize(img, down_size, cv2.INTER_AREA)

                for method in self.upscale_methods:
                    upscaled = cv2.resize(downscaled, (min_dim, min_dim), method)
                    method_name = "cubic" if method == cv2.INTER_CUBIC else "lanczos"

                    output_path = self.output_dir / f"{img_path.stem}_{scale}_{method_name}.jpg"
                    cv2.imwrite(str(output_path), upscaled)

                    # Categorize images based on name
                    img_type = 'other'
                    stem = img_path.stem.lower()
                    for cat, patterns in self.categories.items():
                        if any(p in stem for p in patterns):
                            img_type = cat
                            break

                    results.append({
                        "image": img_path.stem,
                        "category": img_type,
                        "scale": scale,
                        "method": method_name,
                        "psnr": float(psnr(img, upscaled)),
                        "ssim": float(ssim(img, upscaled, channel_axis=2))
                    })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'metrics.csv', index=False)
        return df

    def plot_metrics(self, df):
        plt.figure(figsize=(15, 5))

        # From transcript: Quality plots with confidence intervals
        plt.subplot(1, 3, 1)
        for method in df['method'].unique():
            data = df[df['method'] == method]
            mean = data.groupby('scale')['psnr'].mean()
            std = data.groupby('scale')['psnr'].std()
            plt.plot(mean.index, mean.values, '-o', label=method)
            plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        plt.xlabel('Scale Factor')
        plt.ylabel('PSNR (dB)')
        plt.title('Quality vs Scale')
        plt.legend()
        plt.grid(True)

        # From transcript: "2D chart showing clustering"
        plt.subplot(1, 3, 2)
        for category in df['category'].unique():
            data = df[df['category'] == category]
            plt.scatter(data['scale'], data['psnr'], label=category, alpha=0.6)
        plt.xlabel('Scale Factor')
        plt.ylabel('PSNR')
        plt.title('Content Clustering')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        for category in df['category'].unique():
            data = df[df['category'] == category]
            plt.scatter(data['psnr'], data['ssim'], label=category)
        plt.xlabel('PSNR')
        plt.ylabel('SSIM')
        plt.title('Quality Metrics Clustering')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'analysis.png', dpi=300)
        plt.close()

    def analyze_significance(self, df):
        # From transcript: "student t-test with 95 confidence interval"
        print("\nStatistical Analysis (95% confidence):")

        for category in df['category'].unique():
            print(f"\nCategory: {category}")
            cat_data = df[df['category'] == category]

            # Compare methods
            for method in df['method'].unique():
                method_data = cat_data[cat_data['method'] == method]
                t_stat, p_val = stats.ttest_1samp(method_data['psnr'], 0)
                print(f"{method} PSNR p-value: {p_val:.4f}")


def main():
    processor = ImageProcessor("set14", "processed_images")
    results = processor.process_dataset()
    processor.plot_metrics(results)
    processor.analyze_significance(results)


if __name__ == "__main__":
    main()
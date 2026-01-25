"""
Test Runtime Performance
Measures FPS and memory usage for real-time SLAM requirements.

Target: ≥20 FPS on RTX 5070
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tum_dataset import TUMDataset
from models.descriptor_refiner import DescriptorRefiner
from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector


class PerformanceTester:
    """Test runtime performance and memory usage"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load models
        print("Loading models...")
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['selector_hidden']
        ).to(self.device)

        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])

        self.backbone.eval()
        self.selector.eval()
        self.refiner.eval()

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  Running on CPU (will be very slow!)")

    @torch.no_grad()
    def measure_component_times(self, image: torch.Tensor, num_runs: int = 100):
        """Measure time for each component"""
        times = {
            'backbone': [],
            'selector': [],
            'selector_nms': [],
            'refiner': [],
            'total': []
        }

        # Warmup
        for _ in range(10):
            _ = self.forward_pass(image)

        # Measure
        for _ in range(num_runs):
            # Total time
            t_start = time.perf_counter()

            # Backbone
            t0 = time.perf_counter()
            dino_features = self.backbone(image)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times['backbone'].append((t1 - t0) * 1000)

            # Selector (saliency prediction)
            t0 = time.perf_counter()
            saliency_map = self.selector(dino_features)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times['selector'].append((t1 - t0) * 1000)

            # Selector (keypoint selection with NMS)
            t0 = time.perf_counter()
            keypoints_patch, scores = self.selector.select_keypoints(
                saliency_map,
                num_keypoints=self.config['model']['num_keypoints']
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times['selector_nms'].append((t1 - t0) * 1000)

            # Refiner
            feat_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)
            t0 = time.perf_counter()
            descriptors = self.refiner(feat_at_kpts)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times['refiner'].append((t1 - t0) * 1000)

            # Total
            t_end = time.perf_counter()
            times['total'].append((t_end - t_start) * 1000)

        # Compute statistics
        results = {}
        for key, values in times.items():
            results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        return results

    @torch.no_grad()
    def forward_pass(self, image: torch.Tensor):
        """Complete forward pass"""
        dino_features = self.backbone(image)
        saliency_map = self.selector(dino_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )
        feat_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)
        descriptors = self.refiner(feat_at_kpts)
        return keypoints_patch, descriptors, scores

    def measure_memory_usage(self, image: torch.Tensor):
        """Measure GPU memory usage"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure baseline
        baseline_memory = torch.cuda.memory_allocated() / 1e9

        # Forward pass
        _ = self.forward_pass(image)

        # Measure peak
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9

        return {
            'baseline_gb': baseline_memory,
            'current_gb': current_memory,
            'peak_gb': peak_memory,
            'forward_pass_gb': current_memory - baseline_memory
        }

    def test_sequence(self, sequence: str, num_frames: int = 50):
        """Test performance on a sequence"""
        print(f"\nTesting sequence: {sequence}")

        # Load dataset
        dataset = TUMDataset(
            dataset_root=self.config['dataset']['root'],
            sequence=sequence,
            input_size=self.config['model']['input_size'],
            frame_spacing=1,
            max_frames=num_frames,
            augmentation=None,
            is_train=False
        )

        # Get a sample image
        sample_batch = dataset[0]
        sample_image = sample_batch['rgb1'].unsqueeze(0).to(self.device)

        # Measure component times
        print("  Measuring component times...")
        time_results = self.measure_component_times(sample_image, num_runs=100)

        # Measure memory
        print("  Measuring memory usage...")
        memory_results = self.measure_memory_usage(sample_image)

        # Compute FPS
        total_time_ms = time_results['total']['mean']
        fps = 1000.0 / total_time_ms

        return {
            'sequence': sequence,
            'times': time_results,
            'memory': memory_results,
            'fps': fps
        }

    def visualize_results(self, results: list, output_path: str = 'test/results/performance.png'):
        """Visualize performance results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Component timing breakdown
        ax1 = axes[0, 0]

        # Average across sequences
        components = ['backbone', 'selector', 'selector_nms', 'refiner']
        comp_names = ['DINOv3', 'Selector\n(Saliency)', 'Selector\n(NMS)', 'Refiner']
        means = []
        stds = []

        for comp in components:
            comp_means = [r['times'][comp]['mean'] for r in results]
            means.append(np.mean(comp_means))
            stds.append(np.std(comp_means))

        x = np.arange(len(components))
        bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comp_names)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Component Timing Breakdown')
        ax1.grid(True, alpha=0.3, axis='y')

        # Color bars by relative time
        total_time = sum(means)
        for i, bar in enumerate(bars):
            percentage = means[i] / total_time * 100
            if percentage > 60:
                bar.set_color('red')
            elif percentage > 30:
                bar.set_color('orange')
            else:
                bar.set_color('green')

        # Add percentage labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            percentage = mean / total_time * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 2: FPS by sequence
        ax2 = axes[0, 1]
        sequences = [r['sequence'] for r in results]
        fps_values = [r['fps'] for r in results]

        x = np.arange(len(sequences))
        bars = ax2.bar(x, fps_values, alpha=0.7, edgecolor='black')

        # Color by FPS
        for i, (bar, fps) in enumerate(zip(bars, fps_values)):
            if fps >= 20:
                bar.set_color('green')
            elif fps >= 15:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        ax2.axhline(20, color='darkgreen', linestyle='--', linewidth=2, label='Target: 20 FPS')
        ax2.axhline(30, color='blue', linestyle=':', linewidth=1, label='Excellent: 30 FPS')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax2.set_ylabel('FPS')
        ax2.set_title('Frames Per Second')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add FPS labels on bars
        for i, (bar, fps) in enumerate(zip(bars, fps_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{fps:.1f}', ha='center', va='bottom', fontsize=9)

        # Plot 3: Memory usage
        ax3 = axes[1, 0]

        if torch.cuda.is_available():
            memory_components = ['baseline_gb', 'forward_pass_gb', 'peak_gb']
            comp_names = ['Baseline', 'Forward Pass', 'Peak']
            memory_means = []

            for comp in memory_components:
                mem_values = [r['memory'].get(comp, 0) for r in results]
                memory_means.append(np.mean(mem_values))

            x = np.arange(len(memory_components))
            ax3.bar(x, memory_means, alpha=0.7, edgecolor='black', color=['blue', 'orange', 'red'])
            ax3.set_xticks(x)
            ax3.set_xticklabels(comp_names)
            ax3.set_ylabel('Memory (GB)')
            ax3.set_title('GPU Memory Usage')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, mem in enumerate(memory_means):
                ax3.text(i, mem + 0.05, f'{mem:.2f} GB', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'GPU Not Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.axis('off')

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        avg_fps = np.mean([r['fps'] for r in results])
        avg_total_time = np.mean([r['times']['total']['mean'] for r in results])
        avg_backbone_time = np.mean([r['times']['backbone']['mean'] for r in results])
        avg_selector_time = np.mean([r['times']['selector']['mean'] for r in results])
        avg_refiner_time = np.mean([r['times']['refiner']['mean'] for r in results])

        if torch.cuda.is_available() and results[0]['memory']:
            avg_memory = np.mean([r['memory'].get('forward_pass_gb', 0) for r in results])
        else:
            avg_memory = 0.0

        summary_text = f"""
PERFORMANCE SUMMARY
{'='*45}

Overall Performance:
  Average FPS:      {avg_fps:.1f}
  Total Time:       {avg_total_time:.1f} ms
  Memory Usage:     {avg_memory:.2f} GB

Timing Breakdown:
  DINOv3:           {avg_backbone_time:.1f} ms ({avg_backbone_time/avg_total_time*100:.0f}%)
  Selector:         {avg_selector_time:.1f} ms ({avg_selector_time/avg_total_time*100:.0f}%)
  Refiner:          {avg_refiner_time:.1f} ms ({avg_refiner_time/avg_total_time*100:.0f}%)

Target:             ≥20 FPS
Status:             {'✅ PASS' if avg_fps >= 20 else '❌ FAIL'}

{'='*45}

Real-time Capability:
"""

        if avg_fps >= 30:
            summary_text += "\n  ✅ Excellent! Can run at 30+ FPS"
        elif avg_fps >= 20:
            summary_text += "\n  ✅ Real-time capable (20-30 FPS)"
        elif avg_fps >= 15:
            summary_text += "\n  ⚠️  Borderline (15-20 FPS)"
        else:
            summary_text += "\n  ❌ Too slow for real-time SLAM"

        ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test runtime performance')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml')
    parser.add_argument('--sequences', nargs='+',
                       default=['rgbd_dataset_freiburg1_plant'])
    parser.add_argument('--output', type=str, default='test/results/performance.png')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("RUNTIME PERFORMANCE TEST")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences:  {len(args.sequences)}")
    print("="*70 + "\n")

    tester = PerformanceTester(args.checkpoint, args.config)

    all_results = []
    for seq in args.sequences:
        result = tester.test_sequence(seq, num_frames=50)
        all_results.append(result)

        print(f"\n{'='*70}")
        print(f"Results for {seq}")
        print(f"{'='*70}")
        print(f"FPS:                 {result['fps']:.1f}")
        print(f"Total Time:          {result['times']['total']['mean']:.1f} ms")
        print(f"  DINOv3:            {result['times']['backbone']['mean']:.1f} ms")
        print(f"  Selector:          {result['times']['selector']['mean']:.1f} ms")
        print(f"  Selector (NMS):    {result['times']['selector_nms']['mean']:.1f} ms")
        print(f"  Refiner:           {result['times']['refiner']['mean']:.1f} ms")

        if result['memory']:
            print(f"Memory Usage:        {result['memory']['forward_pass_gb']:.2f} GB")
            print(f"Peak Memory:         {result['memory']['peak_gb']:.2f} GB")

        if result['fps'] >= 20:
            print(f"Status:              ✅ Real-time capable")
        else:
            print(f"Status:              ❌ Too slow (<20 FPS)")
        print(f"{'='*70}")

    # Overall summary
    avg_fps = np.mean([r['fps'] for r in all_results])
    print(f"\n{'='*70}")
    print(f"AVERAGE FPS: {avg_fps:.1f}")
    print(f"{'='*70}")

    if avg_fps >= 20:
        print("✅ PASS: Model meets real-time requirements!")
    else:
        print("❌ FAIL: Too slow for real-time SLAM")
        print("   Consider:")
        print("   - Use smaller DINOv3 model")
        print("   - Reduce number of keypoints")
        print("   - Optimize selector/refiner architecture")

    tester.visualize_results(all_results, args.output)

    # Save results
    import json
    results_json = {
        'average_fps': avg_fps,
        'sequences': []
    }
    for r in all_results:
        results_json['sequences'].append({
            'name': r['sequence'],
            'fps': r['fps'],
            'total_time_ms': r['times']['total']['mean'],
            'memory_gb': r['memory'].get('forward_pass_gb', 0) if r['memory'] else 0
        })

    output_path = Path(args.output)
    json_path = output_path.with_name(f"{output_path.stem}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Saved results to {json_path}")


if __name__ == "__main__":
    main()
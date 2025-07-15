import argparse
import os
from hydra import initialize, compose
from omegaconf import OmegaConf

# Use the main from pred_jointist.py
from pred_jointist import main as run_jointist


def main():
    parser = argparse.ArgumentParser(description="Run Jointist Prediction")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to output .mid file")
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.input.lower().endswith(".wav"):
        raise ValueError("Only .wav files are supported for input")

    # Default MIDI output path if not provided
    output_midi = args.output
    if output_midi is None:
        output_name = os.path.basename(args.input) + ".mid"
        output_midi = os.path.join("MIDI_output", output_name)
    else:
        # Ensure the output directory exists, and convert to absolute path
        output_midi = os.path.abspath(output_midi)
        out_dir = os.path.dirname(output_midi)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Hydra overrides to inject paths
    overrides = [
        f"audio_path={os.path.abspath(args.input)}",
        f"audio_ext=wav",
        f"+output={output_midi}",
    ]

    with initialize(config_path="End2End/config/", version_base=None):
        cfg = compose(config_name="jointist_inference", overrides=overrides)
        run_jointist(cfg)


if __name__ == "__main__":
    main()
import argparse
import soundfile as sf
from lavasr_core import LavaSR


def main():
    parser = argparse.ArgumentParser(
        description="LavaSR runtime-only inference (onnxruntime + numpy + scipy + soundfile)"
    )
    parser.add_argument("input", help="input wav")
    parser.add_argument("-o", "--output", default="output.wav")
    parser.add_argument("--denoise", action="store_true", help="apply ULUNAS denoiser")
    parser.add_argument("--config", default="config.yaml", help="path to enhancer config.yaml")
    parser.add_argument("--denoiser-onnx", default="denoiser_core_legacy_fixed63.onnx", help="path to exported denoiser ONNX")
    parser.add_argument("--enhancer-backbone-onnx", default="enhancer_backbone.onnx", help="path to exported enhancer backbone ONNX")
    parser.add_argument("--enhancer-spec-head-onnx", default="enhancer_spec_head.onnx", help="path to exported enhancer spectrogram-head ONNX")
    parser.add_argument("--ort-provider", default="cpu", choices=["cpu", "cuda"], help="ONNX Runtime execution provider preference")
    parser.add_argument("--ort-intra-threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    parser.add_argument("--ort-inter-threads", type=int, default=1, help="ONNX Runtime inter-op threads")
    args = parser.parse_args()

    ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.ort_provider == "cuda" else ["CPUExecutionProvider"]

    model = LavaSR(
        config=args.config,
        denoiser_onnx=args.denoiser_onnx,
        enhancer_backbone_onnx=args.enhancer_backbone_onnx,
        enhancer_spec_head_onnx=args.enhancer_spec_head_onnx,
        ort_providers=ort_providers,
        ort_intra_op_num_threads=args.ort_intra_threads,
        ort_inter_op_num_threads=args.ort_inter_threads,
    )

    print("Loading audio...")
    wav = model.load_audio(args.input)

    print("Running enhancement...")
    enhanced = model.enhance(wav, apply_denoise=args.denoise)

    sf.write(args.output, enhanced, 48000)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

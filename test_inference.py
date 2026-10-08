"""End-to-end smoke test for the inference pipeline with a verified LoRA."""

import sys
import traceback
from pathlib import Path


def main() -> None:
    print("=" * 50)
    print("LORA INFERENCE PIPELINE TEST")
    print("=" * 50)

    phase = "CONFIG"
    try:
        print("[CONFIG] Creating AppConfig...")
        from src.inference.config import AppConfig, create_default_config

        config: AppConfig = create_default_config()
        if len(sys.argv) != 2:
            raise ValueError("Usage: python test_inference.py <checkpoint.safetensors>")

        checkpoint_path = Path(sys.argv[1])
        if not checkpoint_path.is_absolute():
            checkpoint_path = config.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"LoRA checkpoint does not exist: {checkpoint_path}"
            )
        print(f"[CONFIG] LoRA checkpoint: {checkpoint_path}")
        print("[CONFIG] OK")

        phase = "PIPELINE"
        print("[PIPELINE] Building inference pipeline...")
        from src.inference.pipeline import build_pipeline

        pipe = build_pipeline(config)
        print("[PIPELINE] Pipeline built successfully.")
        print(f"[PIPELINE] Type: {type(pipe).__name__}")
        print(f"[PIPELINE] Scheduler: {type(pipe.scheduler).__name__}")
        print(f"[PIPELINE] Device: {pipe.device}")

        phase = "LORA"
        print("[LORA] Loading and verifying LoRA...")
        from src.inference.lora_loader import load_lora

        lora_result = load_lora(pipe, checkpoint_path)
        print("[LORA] LoRA loaded and verified successfully.")
        print(f"[LORA] Adapter: {lora_result.adapter_name}")

        phase = "GENERATION"
        print("[GENERATION] Starting image generation...") 
        from src.inference.generator import generate_and_save

        saved_paths = generate_and_save(pipe, config.generation)
        print("[GENERATION] Generation completed.")

        phase = "OUTPUT"
        print("[OUTPUT] Verifying saved images...")
        if not saved_paths:
            raise RuntimeError("No generated image paths were returned.")

        expected_dir = (config.project_root / "inference_outputs").resolve()
        for saved_path in saved_paths:
            path = Path(saved_path)
            if not path.is_file():
                raise FileNotFoundError(f"Generated image file does not exist: {path}")
            if path.parent.resolve() != expected_dir or path.suffix.lower() != ".png":
                raise ValueError(f"Generated image is not a PNG in {expected_dir}: {path}")
            print(f"[OUTPUT] Saved: {path.resolve().relative_to(config.project_root)}")

        print(f"[OUTPUT] Verified {len(saved_paths)} image(s).")
        print("[SUCCESS] End-to-end LoRA inference test completed successfully.")
    except Exception as exc:
        print(f"[{phase}] FAILED")
        print(f"[ERROR][{phase}]")
        print(f"Type: {type(exc).__name__}")
        print(f"Message: {exc}")
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

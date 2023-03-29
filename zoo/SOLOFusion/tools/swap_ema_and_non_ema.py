import torch
import sys
import os

def swap_ema_and_non_ema(load_path):
    assert ".pth" in load_path
    assert os.path.exists(load_path), load_path

    ckpt = torch.load(load_path, map_location='cpu')

    # 
    for k in list(ckpt['state_dict'].keys()):
        if k[:4] != "ema_":
            ema_name = f"ema_{k.replace('.', '_')}"
            # Set the original parameter names as the ema parameters
            ckpt['state_dict'][k] = ckpt['state_dict'][ema_name]
            # Remove the ema parameter
            del ckpt['state_dict'][ema_name]

    out_path = load_path.replace(".pth", "_ema.pth")
    if os.path.exists(out_path):
        print(out_path, "exists, not overwriting.")
    else:
        torch.save(ckpt, out_path)

    return out_path

def main():
    assert len(sys.argv) == 2
    assert os.path.exists(sys.argv[1]), sys.argv[1]
    print(swap_ema_and_non_ema(sys.argv[1]))

if __name__ == "__main__":
    main()
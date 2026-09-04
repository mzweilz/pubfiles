import argparse
import hashlib
import sys
from pathlib import Path

import onnx
import torch

WEIGHT_SHA256 = 'e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True, type=Path, help='U-2-Net checkout at the pinned commit')
    parser.add_argument('--weights', required=True, type=Path, help='official u2netp.pth')
    parser.add_argument('--output', default=Path('u2netp-v1.onnx'), type=Path)
    args = parser.parse_args()
    if sha256(args.weights) != WEIGHT_SHA256:
        raise SystemExit('official weight SHA-256 mismatch')

    sys.path.insert(0, str(args.upstream))
    from model.u2net import U2NETP

    class MainMask(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor):
            return self.model(image)[0]

    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    model.eval()
    torch.onnx.export(
        MainMask(model), torch.zeros(1, 3, 320, 320), args.output,
        input_names=['input'], output_names=['mask'], opset_version=11,
        do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.output))
    print(f'{args.output}: {args.output.stat().st_size} bytes, sha256={sha256(args.output)}')


if __name__ == '__main__':
    main()

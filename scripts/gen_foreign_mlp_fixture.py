"""Synthetic MLP fixture: a small net (in=5, hidden=8, out=4, 2 hidden layers) with
LeakyReLU(0.01), torch forward on random inputs -> expected logits. Verifies the
candle forward LOGIC (Linear stack + LeakyReLU), NOT Immortal's weights. Safe to
commit."""
import json, pathlib, torch

torch.manual_seed(7)
IN, HID, OUT, NH = 5, 8, 4, 2
sizes = [IN] + [HID] * NH + [OUT]
lins = [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
sd = {}
for i, lin in enumerate(lins):
    sd[f"net.{2 * i}.weight"] = lin.weight.detach().numpy().tolist()
    sd[f"net.{2 * i}.bias"] = lin.bias.detach().numpy().tolist()


def fwd(x):
    for i, lin in enumerate(lins):
        x = lin(x)
        if i < len(lins) - 1:
            x = torch.nn.functional.leaky_relu(x, 0.01)
    return x


inputs = torch.randn(6, IN)
with torch.no_grad():
    logits = fwd(inputs)
out = pathlib.Path("engine/tests/fixtures/foreign_mlp.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "in_dim": IN, "hidden": HID, "out_dim": OUT, "n_hidden": NH,
    "weights": sd, "inputs": inputs.tolist(), "logits": logits.tolist(),
}))
print(f"wrote {out}")

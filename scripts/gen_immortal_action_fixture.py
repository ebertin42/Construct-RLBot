"""Emit Immortal's 126-row action lookup table as a fixture for the Rust golden
test. Reproduces rlmarlbot/immortal/action/actionparser.py::ImmortalAction
exactly (verified 126x8). Weights-free -- safe to commit."""
import json, pathlib


def make_table():
    actions = []
    # Ground (36 rows)
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial (90 rows)
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if pitch == roll == jump == 0:
                            continue
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 1])
    return actions


def main():
    table = make_table()
    assert len(table) == 126, len(table)
    out = pathlib.Path("engine/tests/fixtures/immortal_action.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table}))
    print(f"wrote {out} ({len(table)} rows)")


if __name__ == "__main__":
    main()

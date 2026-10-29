"""Top-level CLI script to launch the interactive NAYAK visual simulation demo."""

from __future__ import annotations

import argparse
from nayak.demo.app import TrafficDemoApp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch NAYAK: Evolved Fuzzy Traffic Signal Controller Interactive Demo."
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="Nayak-Balanced",
        help="Primary controller name: FixedTime, LongestQueueFirst, HandDesignedFuzzy, Nayak-Fast, Nayak-Fair, Nayak-Rescue, Nayak-Balanced",
    )
    parser.add_argument(
        "--compare-controller",
        type=str,
        default="FixedTime",
        help="Secondary controller for split-screen compare mode (default: FixedTime)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="rush_hour",
        choices=["normal", "rush_hour", "two_wheeler_heavy"],
        help="Traffic scenario: normal, rush_hour, two_wheeler_heavy (default: rush_hour)",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Random arrival seed (default: 1000)")
    parser.add_argument("--compare", action="store_true", help="Launch directly in split-screen compare mode")
    parser.add_argument("--headless", action="store_true", help="Run without opening a GUI window")
    parser.add_argument("--record", type=str, default=None, help="Save recording to output file (e.g. results/demo.gif)")
    parser.add_argument("--record-seconds", type=int, default=10, help="Seconds of simulation to record (default: 10)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    args = parser.parse_args()

    app = TrafficDemoApp(
        controller_name=args.controller,
        compare_controller_name=args.compare_controller,
        scenario_name=args.scenario,
        seed=args.seed,
        compare_mode=args.compare,
        headless=args.headless,
        record_path=args.record,
        fps=args.fps,
        max_record_seconds=args.record_seconds,
    )
    app.run()


if __name__ == "__main__":
    main()

"""Interactive Pygame application supporting single view, split-screen compare mode, and recording."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pygame

# Ensure nayak package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nayak.config import DEFAULT_SIM_CONFIG, SCENARIOS, SimulationConfig
from nayak.controllers.base import BaseController
from nayak.controllers.fixed_time import FixedTimeController
from nayak.controllers.fuzzy_controller import EvolvedFuzzyController, HandDesignedFuzzyController
from nayak.controllers.longest_queue import LongestQueueFirstController
from nayak.demo.renderer import IntersectionRenderer
from nayak.sim.arrivals import ArrivalGenerator
from nayak.sim.intersection import Intersection


class TrafficDemoApp:
    """Main application loop managing simulation steps, user interaction, compare mode, and recording."""

    def __init__(
        self,
        controller_name: str = "Nayak-Balanced",
        compare_controller_name: str = "FixedTime(30s)",
        scenario_name: str = "rush_hour",
        seed: int = 1000,
        compare_mode: bool = False,
        headless: bool = False,
        record_path: Optional[str] = None,
        width: int = 1000,
        height: int = 800,
        fps: int = 30,
        max_record_seconds: int = 15,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.headless = headless
        self.record_path = record_path
        self.max_record_seconds = max_record_seconds

        self.scenario_name = scenario_name
        self.scenario_keys = list(SCENARIOS.keys())
        self.seed = seed
        self.compare_mode = compare_mode

        self.primary_controller_name = controller_name
        self.secondary_controller_name = compare_controller_name

        self.speed_multiplier = 1.0
        self.is_paused = False
        self.sim_clock = 0
        self.step_accumulator = 0.0

        if self.headless or self.record_path:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.display.set_caption("NAYAK: Mixed Indian Traffic Fuzzy Signal Controller Demo")

        if self.headless and not self.record_path:
            self.screen = pygame.Surface((self.width, self.height))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        self.renderer = IntersectionRenderer(self.width, self.height)
        self.renderer_left = IntersectionRenderer(self.width // 2, self.height)
        self.renderer_right = IntersectionRenderer(self.width // 2, self.height)

        self.recorded_frames: List[pygame.Surface] = []

        self._load_controllers()
        self.reset_simulation()

    def _load_controllers(self) -> None:
        """Load evolved genome checkpoints or display instructions."""
        self.controllers: Dict[str, BaseController] = {
            "FixedTime": FixedTimeController(30),
            "LongestQueueFirst": LongestQueueFirstController(2.0),
            "HandDesignedFuzzy": HandDesignedFuzzyController(),
        }

        # Check for evolved representative picks
        results_dir = ROOT_DIR / "results"
        variants = ["Nayak-Fast", "Nayak-Fair", "Nayak-Rescue", "Nayak-Balanced"]
        for v in variants:
            p = results_dir / f"{v}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    g = json.load(f)["genome"]
                self.controllers[v] = EvolvedFuzzyController(g, name=v)
            else:
                print(f"[NOTE] Pretrained genome '{v}.json' not found. Run evolution to enable {v}.")

    def _get_controller(self, name_key: str) -> BaseController:
        """Fetch controller instance matching name query."""
        for k, ctrl in self.controllers.items():
            if name_key.lower() in k.lower():
                return ctrl
        return self.controllers.get("Nayak-Balanced", FixedTimeController(30))

    def reset_simulation(self) -> None:
        """Initialize or reset primary and secondary simulators with identical seeds."""
        self.sim_clock = 0
        self.step_accumulator = 0.0

        scenario = SCENARIOS[self.scenario_name]
        sim_cfg = DEFAULT_SIM_CONFIG
        total_time = 7200

        self.sim_primary = Intersection(timing_config=sim_cfg.signal)
        self.gen_primary = ArrivalGenerator(scenario, sim_cfg, total_time, self.seed)

        self.ctrl_primary = self._get_controller(self.primary_controller_name)
        self.ctrl_primary.reset()

        # Secondary simulator for compare mode
        self.sim_secondary = Intersection(timing_config=sim_cfg.signal)
        self.gen_secondary = ArrivalGenerator(scenario, sim_cfg, total_time, self.seed)

        self.ctrl_secondary = self._get_controller(self.secondary_controller_name)
        self.ctrl_secondary.reset()

        self.renderer = IntersectionRenderer(self.width, self.height)
        self.renderer_left = IntersectionRenderer(self.width // 2, self.height)
        self.renderer_right = IntersectionRenderer(self.width // 2, self.height)

    def handle_input(self, event: pygame.event.Event) -> bool:
        """Process keyboard commands."""
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False

            elif event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused

            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)

            elif event.key == pygame.K_MINUS:
                self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

            elif event.key == pygame.K_r:
                self.reset_simulation()

            elif event.key == pygame.K_n:
                self.seed += 1
                self.reset_simulation()

            elif event.key == pygame.K_s:
                # Cycle scenario
                idx = self.scenario_keys.index(self.scenario_name)
                self.scenario_name = self.scenario_keys[(idx + 1) % len(self.scenario_keys)]
                self.reset_simulation()

            elif event.key == pygame.K_c:
                self.compare_mode = not self.compare_mode
                self.reset_simulation()

            elif event.key == pygame.K_1:
                self.primary_controller_name = "FixedTime"
                self.reset_simulation()

            elif event.key == pygame.K_2:
                self.primary_controller_name = "LongestQueueFirst"
                self.reset_simulation()

            elif event.key == pygame.K_3:
                self.primary_controller_name = "HandDesignedFuzzy"
                self.reset_simulation()

            elif event.key == pygame.K_4:
                self.primary_controller_name = "Nayak-Balanced"
                self.reset_simulation()

            elif event.key == pygame.K_5:
                self.primary_controller_name = "Nayak-Rescue"
                self.reset_simulation()

        return True

    def step_simulation(self) -> None:
        """Advance discrete simulation step by 1 second."""
        self.sim_clock += 1

        # Primary simulator step
        arr_primary = self.gen_primary.generate_step(self.sim_clock)
        self.sim_primary.add_arrivals(arr_primary)

        if self.ctrl_primary.decide(self.sim_primary.get_state()) == 1:
            self.sim_primary.request_phase_switch()
        self.sim_primary.step(self.sim_clock)

        # Secondary simulator step (identical arrivals)
        if self.compare_mode:
            arr_secondary = self.gen_secondary.generate_step(self.sim_clock)
            self.sim_secondary.add_arrivals(arr_secondary)

            if self.ctrl_secondary.decide(self.sim_secondary.get_state()) == 1:
                self.sim_secondary.request_phase_switch()
            self.sim_secondary.step(self.sim_clock)

    def run(self) -> None:
        """Execute interactive event and rendering loop."""
        clock = pygame.time.Clock()
        running = True
        record_frames_target = self.fps * self.max_record_seconds

        print(f"Starting Nayak visual demo [Compare Mode: {self.compare_mode}] ...")

        while running:
            dt = clock.tick(self.fps) / 1000.0

            for event in pygame.event.get():
                if not self.handle_input(event):
                    running = False

            if not self.is_paused:
                self.step_accumulator += dt * self.speed_multiplier
                while self.step_accumulator >= 1.0:
                    self.step_simulation()
                    self.step_accumulator -= 1.0

            # Render
            if not self.compare_mode:
                self.renderer.render_frame(
                    surface=self.screen,
                    sim=self.sim_primary,
                    controller_name=self.ctrl_primary.name,
                    scenario_name=self.scenario_name,
                    seed=self.seed,
                    sim_time=self.sim_clock,
                    speed_multiplier=self.speed_multiplier,
                    is_paused=self.is_paused,
                )
            else:
                # Split Screen Compare Mode
                w_half = self.width // 2
                surf_left = pygame.Surface((w_half, self.height))
                surf_right = pygame.Surface((w_half, self.height))

                self.renderer_left.render_frame(
                    surface=surf_left,
                    sim=self.sim_primary,
                    controller_name=f"[LEFT] {self.ctrl_primary.name}",
                    scenario_name=self.scenario_name,
                    seed=self.seed,
                    sim_time=self.sim_clock,
                    speed_multiplier=self.speed_multiplier,
                    is_paused=self.is_paused,
                )
                self.renderer_right.render_frame(
                    surface=surf_right,
                    sim=self.sim_secondary,
                    controller_name=f"[RIGHT] {self.ctrl_secondary.name}",
                    scenario_name=self.scenario_name,
                    seed=self.seed,
                    sim_time=self.sim_clock,
                    speed_multiplier=self.speed_multiplier,
                    is_paused=self.is_paused,
                )

                self.screen.blit(surf_left, (0, 0))
                self.screen.blit(surf_right, (w_half, 0))

                # Vertical separator
                pygame.draw.line(self.screen, (255, 215, 0), (w_half, 0), (w_half, self.height), 3)

            # Record frames if recording requested
            if self.record_path:
                # Capture frame as numpy array
                frame = pygame.surfarray.array3d(self.screen)
                frame = frame.swapaxes(0, 1)  # (W, H, 3) -> (H, W, 3)
                self.recorded_frames.append(frame)

                if len(self.recorded_frames) >= record_frames_target:
                    self._save_recording()
                    running = False

            if not self.headless:
                pygame.display.flip()

        pygame.quit()

    def _save_recording(self) -> None:
        """Export captured frames to GIF or MP4."""
        if not self.record_path or not self.recorded_frames:
            return

        import imageio
        out_p = Path(self.record_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting {len(self.recorded_frames)} frames to {out_p} ...")
        imageio.mimsave(str(out_p), self.recorded_frames, fps=self.fps)
        print(f"Demo video recording saved successfully to {out_p}")

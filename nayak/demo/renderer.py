"""Pygame 2D renderer for microscopic traffic intersection visualization."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import pygame

from nayak.config import Approach, PhaseState, SignalPhase, VehicleType
from nayak.sim.intersection import Intersection
from nayak.sim.vehicles import Vehicle


# Color Palette
COLOR_GRASS = (28, 38, 30)
COLOR_ASPHALT = (42, 44, 46)
COLOR_ROAD_MARKING_YELLOW = (241, 196, 15)
COLOR_STOP_LINE_WHITE = (245, 245, 245)
COLOR_SIGNAL_RED = (231, 76, 60)
COLOR_SIGNAL_YELLOW = (243, 156, 18)
COLOR_SIGNAL_GREEN = (46, 204, 113)
COLOR_SIGNAL_DARK = (20, 20, 20)
COLOR_HOUSING = (30, 30, 30)

# Vehicle Colors
COLOR_2W = (230, 126, 34)       # Orange
COLOR_AUTO_BODY = (39, 174, 96) # Green
COLOR_AUTO_ROOF = (241, 196, 15)# Yellow
COLOR_CAR = (52, 152, 219)      # Blue
COLOR_BUS = (192, 57, 43)       # Crimson Red
COLOR_AMBULANCE = (255, 255, 255) # White
COLOR_AMB_CROSS = (231, 76, 60) # Red cross
COLOR_BEACON_RED = (255, 20, 20)
COLOR_BEACON_BLUE = (0, 180, 255)

# Dimensions
ROAD_WIDTH = 130
LANE_WIDTH = ROAD_WIDTH // 2


class CrossingAnimation:
    """Animates a vehicle traversing through the intersection after discharge."""

    def __init__(self, vehicle: Vehicle, start_x: float, start_y: float, target_x: float, target_y: float, speed: float = 3.5) -> None:
        self.vehicle = vehicle
        self.x = start_x
        self.y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = speed
        self.finished = False

    def update(self) -> None:
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist <= self.speed:
            self.x = self.target_x
            self.y = self.target_y
            self.finished = True
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed


class IntersectionRenderer:
    """Renders 4-way intersection layout, animated mixed traffic, and live HUD metrics."""

    def __init__(self, width: int = 1000, height: int = 800) -> None:
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2

        self.font_large: Optional[pygame.font.Font] = None
        self.font_medium: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None

        self.crossing_vehicles: List[CrossingAnimation] = []
        self._last_departed_count = 0
        self.frame_counter = 0

    def init_fonts(self) -> None:
        """Initialize Pygame fonts once display is ready."""
        if self.font_large is None:
            self.font_large = pygame.font.SysFont("Helvetica", 18, bold=True)
            self.font_medium = pygame.font.SysFont("Helvetica", 14, bold=True)
            self.font_small = pygame.font.SysFont("Courier", 12)

    def draw_roads(self, surface: pygame.Surface) -> None:
        """Draw intersecting 4-way asphalt roads and road markings."""
        # Horizontal Road (EW)
        h_rect = pygame.Rect(0, self.center_y - ROAD_WIDTH // 2, self.width, ROAD_WIDTH)
        pygame.draw.rect(surface, COLOR_ASPHALT, h_rect)

        # Vertical Road (NS)
        v_rect = pygame.Rect(self.center_x - ROAD_WIDTH // 2, 0, ROAD_WIDTH, self.height)
        pygame.draw.rect(surface, COLOR_ASPHALT, v_rect)

        # Center Dashed Yellow Lines
        # Horizontal center line
        for x in range(0, self.center_x - ROAD_WIDTH // 2, 20):
            pygame.draw.line(surface, COLOR_ROAD_MARKING_YELLOW, (x, self.center_y), (x + 10, self.center_y), 2)
        for x in range(self.center_x + ROAD_WIDTH // 2, self.width, 20):
            pygame.draw.line(surface, COLOR_ROAD_MARKING_YELLOW, (x, self.center_y), (x + 10, self.center_y), 2)

        # Vertical center line
        for y in range(0, self.center_y - ROAD_WIDTH // 2, 20):
            pygame.draw.line(surface, COLOR_ROAD_MARKING_YELLOW, (self.center_x, y), (self.center_x, y + 10), 2)
        for y in range(self.center_y + ROAD_WIDTH // 2, self.height, 20):
            pygame.draw.line(surface, COLOR_ROAD_MARKING_YELLOW, (self.center_x, y), (self.center_x, y + 10), 2)

        # Solid White Stop Lines
        stop_offset = ROAD_WIDTH // 2
        # North stop line (incoming south-bound lane on west side of center line)
        pygame.draw.line(
            surface, COLOR_STOP_LINE_WHITE,
            (self.center_x - ROAD_WIDTH // 2, self.center_y - stop_offset),
            (self.center_x, self.center_y - stop_offset), 4
        )
        # South stop line (incoming north-bound lane on east side of center line)
        pygame.draw.line(
            surface, COLOR_STOP_LINE_WHITE,
            (self.center_x, self.center_y + stop_offset),
            (self.center_x + ROAD_WIDTH // 2, self.center_y + stop_offset), 4
        )
        # East stop line (incoming west-bound lane on north side of center line)
        pygame.draw.line(
            surface, COLOR_STOP_LINE_WHITE,
            (self.center_x + stop_offset, self.center_y - ROAD_WIDTH // 2),
            (self.center_x + stop_offset, self.center_y), 4
        )
        # West stop line (incoming east-bound lane on south side of center line)
        pygame.draw.line(
            surface, COLOR_STOP_LINE_WHITE,
            (self.center_x - stop_offset, self.center_y),
            (self.center_x - stop_offset, self.center_y + ROAD_WIDTH // 2), 4
        )

    def draw_traffic_signals(self, surface: pygame.Surface, sim: Intersection) -> None:
        """Render 3-aspect signal heads for each approach based on intersection state."""
        phase = sim.current_phase
        state = sim.phase_state

        def get_signal_colors(is_active: bool) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
            if is_active:
                if state == PhaseState.GREEN:
                    return COLOR_SIGNAL_DARK, COLOR_SIGNAL_DARK, COLOR_SIGNAL_GREEN
                elif state == PhaseState.YELLOW:
                    return COLOR_SIGNAL_DARK, COLOR_SIGNAL_YELLOW, COLOR_SIGNAL_DARK
                else:  # ALL_RED
                    return COLOR_SIGNAL_RED, COLOR_SIGNAL_DARK, COLOR_SIGNAL_DARK
            else:
                return COLOR_SIGNAL_RED, COLOR_SIGNAL_DARK, COLOR_SIGNAL_DARK

        ns_active = (phase == SignalPhase.PHASE_NS)
        ew_active = (phase == SignalPhase.PHASE_EW)

        signals = [
            (Approach.N, (self.center_x - ROAD_WIDTH // 2 - 25, self.center_y - ROAD_WIDTH // 2 - 50), ns_active, False),
            (Approach.S, (self.center_x + ROAD_WIDTH // 2 + 10, self.center_y + ROAD_WIDTH // 2 + 10), ns_active, False),
            (Approach.E, (self.center_x + ROAD_WIDTH // 2 + 10, self.center_y - ROAD_WIDTH // 2 - 35), ew_active, True),
            (Approach.W, (self.center_x - ROAD_WIDTH // 2 - 50, self.center_y + ROAD_WIDTH // 2 + 10), ew_active, True),
        ]

        radius = 5
        for _, (sx, sy), is_active, is_horizontal in signals:
            r_col, y_col, g_col = get_signal_colors(is_active)
            if not is_horizontal:
                # Vertical housing
                pygame.draw.rect(surface, COLOR_HOUSING, (sx, sy, 16, 42), border_radius=4)
                pygame.draw.circle(surface, r_col, (sx + 8, sy + 8), radius)
                pygame.draw.circle(surface, y_col, (sx + 8, sy + 21), radius)
                pygame.draw.circle(surface, g_col, (sx + 8, sy + 34), radius)
            else:
                # Horizontal housing
                pygame.draw.rect(surface, COLOR_HOUSING, (sx, sy, 42, 16), border_radius=4)
                pygame.draw.circle(surface, r_col, (sx + 8, sy + 8), radius)
                pygame.draw.circle(surface, y_col, (sx + 21, sy + 8), radius)
                pygame.draw.circle(surface, g_col, (sx + 34, sy + 8), radius)

    def draw_vehicle(self, surface: pygame.Surface, veh: Vehicle, x: float, y: float, heading: str) -> None:
        """Render individual vehicle with distinct visual shape, color, and emergency beacons."""
        vtype = veh.vehicle_type

        # Vehicle size definitions: (length, width)
        size_map = {
            VehicleType.TWO_WHEELER: (10, 6),
            VehicleType.AUTO: (18, 11),
            VehicleType.CAR: (26, 13),
            VehicleType.BUS: (48, 16),
            VehicleType.AMBULANCE: (30, 14),
        }
        length, width = size_map[vtype]

        # Orientation alignment
        if heading in ("N", "S"):
            # Vertical travel: height is length, width is width
            w, h = width, length
        else:
            # Horizontal travel: width is length, height is width
            w, h = length, width

        v_rect = pygame.Rect(int(x - w // 2), int(y - h // 2), w, h)

        if vtype == VehicleType.TWO_WHEELER:
            pygame.draw.ellipse(surface, COLOR_2W, v_rect)
            # Rider helmet dot
            pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), 2)

        elif vtype == VehicleType.AUTO:
            pygame.draw.rect(surface, COLOR_AUTO_BODY, v_rect, border_radius=3)
            # Canvas roof overlay
            roof_rect = v_rect.inflate(-3, -3)
            pygame.draw.rect(surface, COLOR_AUTO_ROOF, roof_rect, border_radius=2)

        elif vtype == VehicleType.CAR:
            pygame.draw.rect(surface, COLOR_CAR, v_rect, border_radius=4)
            # Windshield
            if heading in ("N", "S"):
                pygame.draw.rect(surface, (20, 20, 20), (v_rect.x + 2, v_rect.y + h // 3, w - 4, 3))
            else:
                pygame.draw.rect(surface, (20, 20, 20), (v_rect.x + w // 3, v_rect.y + 2, 3, h - 4))

        elif vtype == VehicleType.BUS:
            pygame.draw.rect(surface, COLOR_BUS, v_rect, border_radius=5)
            # Roof ventilation / window trim
            trim_rect = v_rect.inflate(-4, -4)
            pygame.draw.rect(surface, (230, 230, 230), trim_rect, 1)

        elif vtype == VehicleType.AMBULANCE:
            pygame.draw.rect(surface, COLOR_AMBULANCE, v_rect, border_radius=4)
            # Red emergency cross
            cx, cy = int(x), int(y)
            pygame.draw.line(surface, COLOR_AMB_CROSS, (cx - 4, cy), (cx + 4, cy), 2)
            pygame.draw.line(surface, COLOR_AMB_CROSS, (cx, cy - 4), (cx, cy + 4), 2)

            # Flashing dual-color emergency beacon
            beacon_color = COLOR_BEACON_RED if (self.frame_counter // 8) % 2 == 0 else COLOR_BEACON_BLUE
            pygame.draw.circle(surface, beacon_color, (cx, cy), 4)

    def draw_queues(self, surface: pygame.Surface, sim: Intersection) -> None:
        """Position and render queued vehicles along each incoming approach lane."""
        stop_dist = ROAD_WIDTH // 2 + 10

        # North Queue (traveling South, in west lane)
        lane_x = self.center_x - LANE_WIDTH // 2
        curr_y = self.center_y - stop_dist
        for veh in sim.queues[Approach.N]:
            self.draw_vehicle(surface, veh, lane_x, curr_y, heading="S")
            curr_y -= int(veh.pcu * 18 + 5)
            if curr_y < 0:
                break

        # South Queue (traveling North, in east lane)
        lane_x = self.center_x + LANE_WIDTH // 2
        curr_y = self.center_y + stop_dist
        for veh in sim.queues[Approach.S]:
            self.draw_vehicle(surface, veh, lane_x, curr_y, heading="N")
            curr_y += int(veh.pcu * 18 + 5)
            if curr_y > self.height:
                break

        # East Queue (traveling West, in north lane)
        lane_y = self.center_y - LANE_WIDTH // 2
        curr_x = self.center_x + stop_dist
        for veh in sim.queues[Approach.E]:
            self.draw_vehicle(surface, veh, curr_x, lane_y, heading="W")
            curr_x += int(veh.pcu * 18 + 5)
            if curr_x > self.width:
                break

        # West Queue (traveling East, in south lane)
        lane_y = self.center_y + LANE_WIDTH // 2
        curr_x = self.center_x - stop_dist
        for veh in sim.queues[Approach.W]:
            self.draw_vehicle(surface, veh, curr_x, lane_y, heading="E")
            curr_x -= int(veh.pcu * 18 + 5)
            if curr_x < 0:
                break

    def track_discharged_vehicles(self, sim: Intersection) -> None:
        """Detect newly discharged vehicles and spawn intersection crossing animations."""
        curr_departed = len(sim.all_departed_vehicles)
        if curr_departed > self._last_departed_count:
            new_departures = sim.all_departed_vehicles[self._last_departed_count : curr_departed]
            self._last_departed_count = curr_departed

            stop_offset = ROAD_WIDTH // 2

            for veh in new_departures:
                if veh.approach == Approach.N:
                    sx = self.center_x - LANE_WIDTH // 2
                    sy = self.center_y - stop_offset
                    tx, ty = sx, self.center_y + stop_offset + 60
                elif veh.approach == Approach.S:
                    sx = self.center_x + LANE_WIDTH // 2
                    sy = self.center_y + stop_offset
                    tx, ty = sx, self.center_y - stop_offset - 60
                elif veh.approach == Approach.E:
                    sx = self.center_x + stop_offset
                    sy = self.center_y - LANE_WIDTH // 2
                    tx, ty = self.center_x - stop_offset - 60, sy
                else:  # Approach.W
                    sx = self.center_x - stop_offset
                    sy = self.center_y + LANE_WIDTH // 2
                    tx, ty = self.center_x + stop_offset + 60, sy

                anim = CrossingAnimation(veh, sx, sy, tx, ty)
                self.crossing_vehicles.append(anim)

    def draw_crossing_vehicles(self, surface: pygame.Surface) -> None:
        """Update and draw vehicles currently crossing the intersection."""
        active = []
        for anim in self.crossing_vehicles:
            anim.update()
            heading = "S" if anim.vehicle.approach == Approach.N else "N" if anim.vehicle.approach == Approach.S else "W" if anim.vehicle.approach == Approach.E else "E"
            self.draw_vehicle(surface, anim.vehicle, anim.x, anim.y, heading)
            if not anim.finished:
                active.append(anim)
        self.crossing_vehicles = active

    def draw_hud(
        self,
        surface: pygame.Surface,
        controller_name: str,
        scenario_name: str,
        seed: int,
        sim_time: int,
        sim: Intersection,
        speed_multiplier: float,
        is_paused: bool,
    ) -> None:
        """Render HUD metrics, signal timings, and keyboard shortcuts."""
        self.init_fonts()

        # Top Banner
        hud_bg = pygame.Surface((self.width, 85))
        hud_bg.set_alpha(220)
        hud_bg.fill((15, 20, 25))
        surface.blit(hud_bg, (0, 0))

        # Title / Controller / Scenario
        ctrl_text = self.font_large.render(f"Controller: {controller_name}", True, (240, 240, 240))
        sc_text = self.font_medium.render(f"Scenario: {scenario_name.upper()} | Seed: {seed}", True, (180, 200, 210))
        time_text = self.font_large.render(f"Sim Time: {sim_time}s", True, (255, 215, 0))

        surface.blit(ctrl_text, (20, 10))
        surface.blit(sc_text, (20, 35))
        surface.blit(time_text, (20, 58))

        # Signal State
        p_name = "PHASE_NS (N/S Green)" if sim.current_phase == SignalPhase.PHASE_NS else "PHASE_EW (E/W Green)"
        sig_info = self.font_medium.render(
            f"Signal: {p_name} | State: {sim.phase_state.value.upper()} ({sim.green_elapsed}s)",
            True,
            COLOR_SIGNAL_GREEN if sim.phase_state == PhaseState.GREEN else COLOR_SIGNAL_YELLOW if sim.phase_state == PhaseState.YELLOW else COLOR_SIGNAL_RED,
        )
        surface.blit(sig_info, (self.width // 2 - 120, 12))

        # Queue Status in PCU
        st = sim.get_state()
        q_info = self.font_small.render(
            f"Queues (PCU): N={st.queue_pcu[Approach.N]:.1f} | S={st.queue_pcu[Approach.S]:.1f} | "
            f"E={st.queue_pcu[Approach.E]:.1f} | W={st.queue_pcu[Approach.W]:.1f}",
            True,
            (210, 210, 210),
        )
        surface.blit(q_info, (self.width // 2 - 120, 36))

        # Throughput & Ambulance Metrics
        amb_count = sum(1 for v in sim.all_departed_vehicles if v.is_ambulance)
        metrics_info = self.font_small.render(
            f"Throughput: {len(sim.all_departed_vehicles)} veh | Ambulances Cleared: {amb_count}",
            True,
            (140, 230, 160),
        )
        surface.blit(metrics_info, (self.width // 2 - 120, 58))

        # Controls Hint (Right Corner)
        paused_str = " [PAUSED]" if is_paused else ""
        speed_text = self.font_medium.render(f"Speed: {speed_multiplier:.1f}x{paused_str}", True, (255, 180, 0))
        surface.blit(speed_text, (self.width - 170, 12))

        help_text = self.font_small.render("1-5:Ctrl S:Scen N:Seed C:Cmp", True, (160, 160, 160))
        help_text2 = self.font_small.render("+/-:Spd Space:Pause R:Reset", True, (160, 160, 160))
        surface.blit(help_text, (self.width - 200, 36))
        surface.blit(help_text2, (self.width - 200, 56))

    def render_frame(
        self,
        surface: pygame.Surface,
        sim: Intersection,
        controller_name: str,
        scenario_name: str,
        seed: int,
        sim_time: int,
        speed_multiplier: float,
        is_paused: bool,
    ) -> None:
        """Render complete animation frame onto the target surface."""
        self.frame_counter += 1
        surface.fill(COLOR_GRASS)

        self.draw_roads(surface)
        self.draw_traffic_signals(surface, sim)
        self.draw_queues(surface, sim)

        self.track_discharged_vehicles(sim)
        self.draw_crossing_vehicles(surface)

        self.draw_hud(
            surface,
            controller_name=controller_name,
            scenario_name=scenario_name,
            seed=seed,
            sim_time=sim_time,
            sim=sim,
            speed_multiplier=speed_multiplier,
            is_paused=is_paused,
        )

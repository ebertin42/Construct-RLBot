use crate::{
    actions,
    obs::{self, OBS_SIZE},
    reward::{self, RewardConfig},
    schema::Normalization,
};
use cxx::UniquePtr;
use rocketsim_rs::{
    math::Vec3,
    sim::{Arena, CarConfig, Team},
    GameState,
};

const TICKS_PER_SEC: u64 = 120;
const NO_TOUCH_TICKS: u64 = 30 * TICKS_PER_SEC;
const MAX_TICKS: u64 = 300 * TICKS_PER_SEC;

#[derive(Debug, Default, Clone, Copy)]
pub struct StepFlags {
    pub terminated: bool,
    pub truncated: bool,
}

pub struct EpisodeArena {
    arena: UniquePtr<Arena>,
    table: Vec<[f32; 8]>,
    car_ids: Vec<u32>, // blue asc, then orange asc — agent index order
    tick_skip: u32,
    reward_cfg: RewardConfig,
    norm: Normalization,
    seed: u32,
    episode_start_tick: u64,
    last_touch_tick: u64,
    prev_state: GameState,
}

impl EpisodeArena {
    pub fn new(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
    ) -> Self {
        let mut arena = Arena::default_standard();
        let mut car_ids = Vec::with_capacity(blue + orange);
        for _ in 0..blue {
            car_ids.push(arena.pin_mut().add_car(Team::Blue, CarConfig::octane()));
        }
        for _ in 0..orange {
            car_ids.push(arena.pin_mut().add_car(Team::Orange, CarConfig::octane()));
        }
        arena.pin_mut().reset_to_random_kickoff(Some(seed));
        let prev_state = arena.pin_mut().get_game_state();
        let start = prev_state.tick_count;
        Self {
            arena,
            table: actions::make_lookup_table(),
            car_ids,
            tick_skip,
            reward_cfg,
            norm,
            seed,
            episode_start_tick: start,
            last_touch_tick: start,
            prev_state,
        }
    }

    pub fn num_agents(&self) -> usize {
        self.car_ids.len()
    }

    fn agent_car_index(&self, state: &GameState, agent: usize) -> usize {
        let id = self.car_ids[agent];
        state.cars.iter().position(|c| c.id == id).expect("car exists")
    }

    pub fn write_obs(&mut self, out: &mut [f32]) {
        let gs = self.arena.pin_mut().get_game_state();
        for a in 0..self.num_agents() {
            let ci = self.agent_car_index(&gs, a);
            obs::build_obs(&gs, ci, &self.norm, &mut out[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
        }
    }

    pub fn step(
        &mut self,
        action_idx: &[i64],
        rewards: &mut [f32],
        flags: &mut [StepFlags],
        final_obs: &mut [f32],
    ) {
        let n = self.num_agents();
        assert_eq!(action_idx.len(), n);

        let controls: Vec<(u32, rocketsim_rs::sim::CarControls)> = (0..n)
            .map(|a| {
                let row = &self.table[action_idx[a] as usize];
                (self.car_ids[a], actions::to_controls(row))
            })
            .collect();
        self.arena.pin_mut().set_all_controls(&controls).expect("valid car ids");
        self.arena.pin_mut().step(self.tick_skip);

        let cur = self.arena.pin_mut().get_game_state();
        let scored = if self.arena.is_ball_scored() {
            Some(if cur.ball.pos.y > 0.0 { Team::Blue } else { Team::Orange })
        } else {
            None
        };

        // Touch tracking for no-touch truncation. Mirrors reward.rs's touch-detection
        // contract: a car that has never touched the ball has `is_valid == false` and
        // `tick_count_when_hit == u64::MAX` (a sentinel, not 0 — see reward.rs's module
        // doc comment for how this was verified). Requiring `is_valid` here is belt-and-
        // suspenders: the `hit <= cur.tick_count` bound already excludes the u64::MAX
        // sentinel in practice, but gating on `is_valid` makes the "never touched" case
        // explicit and keeps this loop consistent with reward.rs's touch check rather
        // than relying solely on the numeric bound.
        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            let hit_info = &cur.cars[ci].state.ball_hit_info;
            let hit = hit_info.tick_count_when_hit;
            if hit_info.is_valid && hit > self.last_touch_tick && hit <= cur.tick_count {
                self.last_touch_tick = hit;
            }
        }

        let terminated = scored.is_some();
        let truncated = !terminated
            && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                || cur.tick_count - self.episode_start_tick >= MAX_TICKS);

        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            rewards[a] = reward::compute(&self.prev_state, &cur, ci, scored, &self.reward_cfg);
            flags[a] = StepFlags { terminated, truncated };
        }

        if terminated || truncated {
            // capture final obs, then reset
            for a in 0..n {
                let ci = self.agent_car_index(&cur, a);
                obs::build_obs(&cur, ci, &self.norm, &mut final_obs[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
            }
            self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
            self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
            let gs = self.arena.pin_mut().get_game_state();
            self.episode_start_tick = gs.tick_count;
            self.last_touch_tick = gs.tick_count;
            self.prev_state = gs;
        } else {
            self.prev_state = cur;
        }
    }

    pub fn game_state(&mut self) -> GameState {
        self.arena.pin_mut().get_game_state()
    }

    /// Test/debug helper: warp the ball.
    pub fn debug_place_ball(&mut self, pos: [f32; 3], vel: [f32; 3]) {
        let mut ball = self.arena.pin_mut().get_ball();
        ball.pos = Vec3::new(pos[0], pos[1], pos[2]);
        ball.vel = Vec3::new(vel[0], vel[1], vel[2]);
        self.arena.pin_mut().set_ball(ball);
    }
}

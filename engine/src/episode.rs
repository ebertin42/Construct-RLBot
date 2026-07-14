use crate::{
    actions,
    curriculum::CurriculumConfig,
    obs::{self, OBS_SIZE},
    reward::{self, RewardConfig},
    sampler::Pcg32,
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

/// r_i' = (1-tau)*r_i + tau*mean(own team) - opp*mean(opponent team).
/// `blue_count` splits `rewards` (blue-then-orange agent order). An empty
/// team's mean is 0.0 (1v0 test arenas).
pub fn blend_team_spirit(rewards: &mut [f32], blue_count: usize, tau: f32, opp: f32) {
    let mean = |s: &[f32]| if s.is_empty() { 0.0 } else { s.iter().sum::<f32>() / s.len() as f32 };
    let bm = mean(&rewards[..blue_count]);
    let om = mean(&rewards[blue_count..]);
    for (i, r) in rewards.iter_mut().enumerate() {
        let (own, other) = if i < blue_count { (bm, om) } else { (om, bm) };
        *r = (1.0 - tau) * *r + tau * own - opp * other;
    }
}

fn vec3_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// True iff every physics quantity the obs/reward paths consume is finite.
fn state_is_finite(gs: &GameState) -> bool {
    if !(vec3_finite(&gs.ball.pos) && vec3_finite(&gs.ball.vel) && vec3_finite(&gs.ball.ang_vel)) {
        return false;
    }
    gs.cars.iter().all(|c| {
        vec3_finite(&c.state.pos)
            && vec3_finite(&c.state.vel)
            && vec3_finite(&c.state.ang_vel)
            && vec3_finite(&c.state.rot_mat.forward)
            && vec3_finite(&c.state.rot_mat.up)
    })
}

pub struct EpisodeArena {
    arena: UniquePtr<Arena>,
    table: Vec<[f32; 8]>,
    car_ids: Vec<u32>, // blue asc, then orange asc — agent index order
    blue_count: usize,
    tick_skip: u32,
    reward_cfg: RewardConfig,
    norm: Normalization,
    seed: u32,
    episode_start_tick: u64,
    last_touch_tick: u64,
    prev_state: GameState,
    curriculum: Option<CurriculumConfig>,
    rng: Pcg32,
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
        Self::new_with_curriculum(blue, orange, tick_skip, reward_cfg, norm, seed, None)
    }

    pub fn new_with_curriculum(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
        curriculum: Option<CurriculumConfig>,
    ) -> Self {
        let mut arena = Arena::default_standard();
        let mut car_ids = Vec::with_capacity(blue + orange);
        for _ in 0..blue {
            car_ids.push(arena.pin_mut().add_car(Team::Blue, CarConfig::octane()));
        }
        for _ in 0..orange {
            car_ids.push(arena.pin_mut().add_car(Team::Orange, CarConfig::octane()));
        }
        // Placeholder state — immediately overwritten by reset_episode() below,
        // which performs the actual (possibly curriculum-driven) reset.
        let prev_state = arena.pin_mut().get_game_state();
        let start = prev_state.tick_count;
        let mut this = Self {
            arena,
            table: actions::make_lookup_table(),
            car_ids,
            blue_count: blue,
            tick_skip,
            reward_cfg,
            norm,
            seed,
            episode_start_tick: start,
            last_touch_tick: start,
            prev_state,
            curriculum,
            rng: Pcg32::new((seed as u64) * 7919 + 13),
        };
        this.reset_episode();
        this
    }

    /// Kickoff when curriculum is absent, or when the weighted coin says kickoff;
    /// otherwise a bounded random scenario. Replaces every raw
    /// `reset_to_random_kickoff` call site: always kicks off first for a clean
    /// baseline (boost pads, `ball_hit_info` reset), then overwrites with a random
    /// scenario when the curriculum draw says so — this keeps pad/hit-info hygiene
    /// identical across both reset flavors.
    fn reset_episode(&mut self) {
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        let use_random = match &self.curriculum {
            Some(c) => {
                let p = c.random_weight / (c.random_weight + c.kickoff_weight);
                self.rng.next_f32() < p
            }
            None => false,
        };
        self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
        if use_random {
            let bounds = self.curriculum.as_ref().unwrap().random.clone();
            crate::curriculum::random_reset(self.arena.pin_mut(), &mut self.rng, &bounds);
        }
        let gs = self.arena.pin_mut().get_game_state();
        self.episode_start_tick = gs.tick_count;
        self.last_touch_tick = gs.tick_count;
        self.prev_state = gs;
    }

    /// Test/debug helper: force an episode reset (through the same curriculum-aware
    /// path `step` uses on termination/truncation).
    pub fn debug_force_reset(&mut self) {
        self.reset_episode();
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

        // Physics-blowup containment. RocketSim's contact solver can go nonfinite
        // in degenerate multi-body squeezes (observed live at 2.2B steps: two
        // near-static cars pinching the ball -> a car ejected to [nan,-inf,nan]).
        // A poisoned state must never reach rewards, obs, or the learner: end the
        // episode with finite zeros and a fresh kickoff. Terminated (not truncated)
        // so GAE bootstraps 0 instead of a value estimate of garbage.
        if !state_is_finite(&cur) {
            eprintln!(
                "[construct-engine] physics blowup contained (tick {}): episode terminated, arena reset",
                cur.tick_count
            );
            for a in 0..n {
                rewards[a] = 0.0;
                flags[a] = StepFlags { terminated: true, truncated: false };
            }
            final_obs[..n * OBS_SIZE].fill(0.0);
            self.reset_episode();
            return;
        }

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

        if self.reward_cfg.team_spirit != 0.0 || self.reward_cfg.opp_spirit != 0.0 {
            blend_team_spirit(
                &mut rewards[..n],
                self.blue_count,
                self.reward_cfg.team_spirit,
                self.reward_cfg.opp_spirit,
            );
        }

        if terminated || truncated {
            // capture final obs, then reset
            for a in 0..n {
                let ci = self.agent_car_index(&cur, a);
                obs::build_obs(&cur, ci, &self.norm, &mut final_obs[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
            }
            self.reset_episode();
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

    /// JSON state dump matching the deploy/obs.py dict contract (Task 12 interfaces).
    pub fn debug_state_json(&mut self) -> String {
        let gs = self.arena.pin_mut().get_game_state();
        let v3 = |v: &Vec3| serde_json::json!([v.x, v.y, v.z]);
        let cars: Vec<serde_json::Value> = self
            .car_ids
            .iter()
            .map(|&id| {
                let c = gs.cars.iter().find(|c| c.id == id).expect("car exists");
                serde_json::json!({
                    "id": c.id,
                    "team": c.team as u8,
                    "pos": v3(&c.state.pos),
                    "vel": v3(&c.state.vel),
                    "ang_vel": v3(&c.state.ang_vel),
                    "forward": v3(&c.state.rot_mat.forward),
                    "up": v3(&c.state.rot_mat.up),
                    "boost": c.state.boost,
                    "is_on_ground": c.state.is_on_ground,
                    "has_flip": c.state.has_flip_or_jump(),
                    "is_demoed": c.state.is_demoed,
                })
            })
            .collect();
        serde_json::json!({
            "ball": {
                "pos": v3(&gs.ball.pos),
                "vel": v3(&gs.ball.vel),
                "ang_vel": v3(&gs.ball.ang_vel),
            },
            "cars": cars,
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_math_exact() {
        // raw: blue [1.0, 3.0], orange [-2.0, 0.0]; tau=0.5, opp=0.25
        // bm=2.0, om=-1.0
        // blue0' = 0.5*1.0 + 0.5*2.0 - 0.25*(-1.0) = 1.75
        // blue1' = 0.5*3.0 + 0.5*2.0 - 0.25*(-1.0) = 2.75
        // org0'  = 0.5*(-2.0) + 0.5*(-1.0) - 0.25*2.0 = -2.0
        // org1'  = 0.5*0.0 + 0.5*(-1.0) - 0.25*2.0 = -1.0
        let mut r = vec![1.0f32, 3.0, -2.0, 0.0];
        blend_team_spirit(&mut r, 2, 0.5, 0.25);
        assert_eq!(r, vec![1.75, 2.75, -2.0, -1.0]);
    }

    #[test]
    fn blend_handles_empty_orange() {
        let mut r = vec![2.0f32];
        blend_team_spirit(&mut r, 1, 0.5, 0.25); // opp mean = 0.0
        assert_eq!(r, vec![2.0]); // (1-.5)*2 + .5*2 - .25*0 = 2.0
    }
}

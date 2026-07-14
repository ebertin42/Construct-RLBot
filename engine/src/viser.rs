use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage};
use std::net::UdpSocket;
use std::time::{Duration, Instant};

pub struct ViserStream {
    socket: UdpSocket,
    codec: PacketCodec,
    target: String,
}

impl ViserStream {
    pub fn new() -> std::io::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", 34254))?;
        socket.set_nonblocking(true)?;
        // Override when rlviser runs on another host (e.g. Windows host from WSL2 NAT):
        // CONSTRUCT_VISER_ADDR=<ip>:45243
        let target = std::env::var("CONSTRUCT_VISER_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:45243".to_string());
        let mut s = Self { socket, codec: PacketCodec::new(), target };
        s.send(RlviserMessage::Connection)?;
        Ok(s)
    }

    pub fn send(&mut self, msg: RlviserMessage) -> std::io::Result<()> {
        let bytes = self.codec.encode(msg);
        self.socket.send_to(bytes, self.target.as_str())?;
        Ok(())
    }

    pub fn send_state(&mut self, gs: rocketsim_rs::GameState) -> std::io::Result<()> {
        self.send(RlviserMessage::GameState(Box::new(gs)))
    }

    pub fn quit(&mut self) {
        let _ = self.send(RlviserMessage::Quit);
    }
}

/// Realtime pacing helper: sleep so each tick_skip step takes tick_skip/120 s.
pub struct Pacer {
    next: Instant,
    step_dur: Duration,
}

impl Pacer {
    pub fn new(tick_skip: u32) -> Self {
        Self { next: Instant::now(), step_dur: Duration::from_secs_f64(tick_skip as f64 / 120.0) }
    }
    pub fn pace(&mut self) {
        let now = Instant::now();
        if self.next > now {
            std::thread::sleep(self.next - now);
        }
        self.next = Instant::now().max(self.next) + self.step_dur;
    }
}

// Probe: send Connection to rlviser, report whether it replies (proves it reached
// its packet-consuming state).
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage, PACKET_SIZE_BYTES};
use std::net::UdpSocket;
use std::time::Duration;

fn main() {
    let socket = UdpSocket::bind(("0.0.0.0", 34254)).expect("bind 34254");
    socket.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
    let target = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:45243".to_string());
    let mut codec = PacketCodec::new();
    let bytes = codec.encode(RlviserMessage::Connection);
    socket.send_to(bytes, target.as_str()).expect("send");
    println!("sent Connection ({} bytes) to {target}", bytes.len());
    let mut buf = vec![0u8; 65536];
    for _ in 0..2 {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                match PacketCodec::decode_payload(&buf[PACKET_SIZE_BYTES..n]) {
                    Ok(Some(msg)) => println!("REPLY from {from}: {} bytes -> {:?}", n, msg_name(&msg)),
                    other => println!("REPLY from {from}: {} bytes -> undecodable: {other:?}", n),
                }
            }
            Err(e) => { println!("NO REPLY within timeout: {e}"); break; }
        }
    }
}

fn msg_name(m: &RlviserMessage) -> &'static str {
    match m {
        RlviserMessage::Connection => "Connection",
        RlviserMessage::Quit => "Quit",
        RlviserMessage::Speed(_) => "Speed",
        RlviserMessage::Paused(_) => "Paused",
        RlviserMessage::GameState(_) => "GameState",
        _ => "Other",
    }
}

// Probe: bind rlviser's port, decode incoming stream packets exactly as rlviser does.
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage, PACKET_SIZE_BYTES};
use std::net::UdpSocket;

fn main() {
    let socket = UdpSocket::bind(("0.0.0.0", 45243)).expect("bind 45243");
    let mut buf = vec![0u8; 65536];
    for i in 0..5 {
        let (n, from) = socket.recv_from(&mut buf).expect("recv");
        let header = u64::from_be_bytes(buf[..8].try_into().unwrap());
        print!("pkt {i}: {n} bytes from {from}, header={header} ");
        match PacketCodec::decode_payload(&buf[PACKET_SIZE_BYTES..n]) {
            Ok(Some(RlviserMessage::GameState(gs))) => {
                println!("-> GameState: {} cars, ball z={:.1}, tick={}", gs.cars.len(), gs.ball.pos.z, gs.tick_count)
            }
            Ok(Some(other)) => println!("-> {:?}", std::mem::discriminant(&other)),
            Ok(None) => println!("-> decoded None"),
            Err(e) => println!("-> DECODE ERROR: {e:?}"),
        }
    }
}

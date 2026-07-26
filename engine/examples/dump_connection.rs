// Writes a framed Connection packet to conn.bin for cross-host testing.
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage};

fn main() {
    let mut codec = PacketCodec::new();
    let bytes = codec.encode(RlviserMessage::Connection);
    std::fs::write("conn.bin", bytes).unwrap();
    println!("wrote conn.bin ({} bytes)", bytes.len());
}

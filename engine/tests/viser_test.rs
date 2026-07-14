use construct_engine::sim_init::ensure_init;
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage, PACKET_SIZE_BYTES};
use rocketsim_rs::sim::{Arena, CarConfig, Team};

#[test]
fn gamestate_packet_roundtrips_through_codec() {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(3));
    let gs = arena.pin_mut().get_game_state();

    let mut codec = PacketCodec::new();
    let bytes = codec.encode(RlviserMessage::GameState(Box::new(gs.clone()))).to_vec();
    assert!(bytes.len() > PACKET_SIZE_BYTES);
    let decoded = PacketCodec::decode_payload(&bytes[PACKET_SIZE_BYTES..]).unwrap().unwrap();
    match decoded {
        RlviserMessage::GameState(g) => {
            assert_eq!(g.cars.len(), 1);
            assert_eq!(g.tick_count, gs.tick_count);
        }
        other => panic!("wrong message decoded: {other:?}"),
    }
}

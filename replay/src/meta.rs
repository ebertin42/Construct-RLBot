use boxcars::{HeaderProp, ParserBuilder};

#[derive(Debug, Clone)]
pub struct ReplayMeta {
    pub playlist: String,
    pub team_size: u8,
    pub duration_secs: u32,
    pub net_version: i32,
    pub num_frames: usize,
}

fn prop_i32(props: &[(String, HeaderProp)], key: &str) -> Option<i32> {
    props.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        HeaderProp::Int(i) => Some(*i),
        HeaderProp::Float(f) => Some(*f as i32),
        _ => None,
    })
}

fn prop_f32(props: &[(String, HeaderProp)], key: &str) -> Option<f32> {
    props.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        HeaderProp::Float(f) => Some(*f),
        HeaderProp::Int(i) => Some(*i as f32),
        _ => None,
    })
}

fn prop_str(props: &[(String, HeaderProp)], key: &str) -> Option<String> {
    props.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        HeaderProp::Str(s) => Some(s.clone()),
        HeaderProp::Name(s) => Some(s.clone()),
        _ => None,
    })
}

/// Parses just the boxcars header (still requires network data to count real
/// simulation frames; boxcars' `NumFrames` header prop is used as a fallback
/// when network frame decoding is unavailable).
///
/// Verified header prop keys against a real ranked-duels fixture replay
/// (`grand-champion-1/duels/batch_0000/...` from the `chrisrca/rocket-league-replays`
/// HF dataset):
///   TeamSize    = Int      (players per team, e.g. 1 for duels)
///   RecordFPS   = Float    (usually 30.0)
///   MatchType   = Name     (e.g. "Online", not a rank/playlist string)
///   NumFrames   = Int      (header-reported frame count; matches network_frames.len())
pub fn parse_meta(bytes: &[u8]) -> Result<ReplayMeta, String> {
    let replay = ParserBuilder::new(bytes)
        .must_parse_network_data()
        .parse()
        .map_err(|e| format!("parse: {e}"))?;
    let props = &replay.properties;
    let team_size = prop_i32(props, "TeamSize").unwrap_or(0) as u8;
    let record_fps = prop_f32(props, "RecordFPS").unwrap_or(30.0).max(1.0);
    let frames = replay
        .network_frames
        .as_ref()
        .map(|nf| nf.frames.len())
        .unwrap_or_else(|| prop_i32(props, "NumFrames").unwrap_or(0) as usize);
    Ok(ReplayMeta {
        playlist: prop_str(props, "MatchType").unwrap_or_else(|| "unknown".into()),
        team_size,
        duration_secs: (frames as f32 / record_fps) as u32,
        net_version: replay.net_version.unwrap_or(0),
        num_frames: frames,
    })
}

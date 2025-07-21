use std::collections::HashMap;

fn get_stats(ids: &[u16]) -> Option<(u16, u16)> {
    if ids.len() < 2 {
        return None;
    }

    let mut counts = HashMap::new();
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(pair, _)| pair)
}

fn merge_bytes(ids: &[u16], pair: (u16, u16), idx: u16) -> Vec<u16> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    new_ids
}

pub fn encode(sentence: &str, num_merges: u16) -> (Vec<u16>, Vec<Vec<u8>>) {
    let mut ids: Vec<u16> = sentence.bytes().map(|b| b as u16).collect();
    let mut vocab: Vec<Vec<u8>> = (0..=255).map(|i| vec![i as u8]).collect();

    for i in 0..num_merges {
        if let Some(pair) = get_stats(&ids) {
            let idx = (256 + i) as u16;
            ids = merge_bytes(&ids, pair, idx);

            let mut new_vocab_entry = vocab[pair.0 as usize].clone();
            new_vocab_entry.extend_from_slice(&vocab[pair.1 as usize]);
            vocab.push(new_vocab_entry);
        } else {
            break;
        }
    }

    (ids, vocab)
}

pub fn decode(byte_seq: &[u16], vocab: Vec<Vec<u8>>) -> String {
    let mut decoded = Vec::new();
    for byte in byte_seq {
        decoded.extend(vocab[*byte as usize].clone());
    }

    String::from_utf8(decoded).unwrap()
}

mod bpe;

fn main() {
    let sentence = "hell12342342o world";
    let num_merges = 10;
    let (encoded, vocab) = bpe::encode(sentence, num_merges);

    println!("Encoded: {:?}", encoded);
    println!("Vocabulary size: {}", vocab.len());

    let decoded = bpe::decode(&encoded, vocab);
    println!("Decoded: {}", decoded);

    assert_eq!(sentence, decoded);
}

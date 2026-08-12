from bbrt.data.tokenizer import SelfiesTokenizer, split_selfies


def test_split_bracket_tokens():
    assert split_selfies("[C][N][=O]") == ["[C]", "[N]", "[=O]"]
    # already space-separated on disk
    assert split_selfies("[C] [N] [=O]") == ["[C]", "[N]", "[=O]"]
    assert split_selfies("") == []


def test_build_and_roundtrip():
    corpus = ["[C][N][=O]", "[C][C][Branch1]"]
    tok = SelfiesTokenizer.build([corpus])
    for special in SelfiesTokenizer.SPECIALS:
        assert special in tok.stoi
    ids = tok.encode("[C][N][=O]")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "[C][N][=O]"


def test_unknown_token_maps_to_unk():
    tok = SelfiesTokenizer.build([["[C][N]"]])
    ids = tok.encode("[C][Xe]", add_bos=False, add_eos=False)
    assert ids[1] == tok.unk_id


def test_decode_stops_at_eos():
    tok = SelfiesTokenizer.build([["[C][N]"]])
    ids = [tok.bos_id, tok.stoi["[C]"], tok.eos_id, tok.stoi["[N]"]]
    assert tok.decode(ids) == "[C]"


def test_save_load(tmp_path):
    tok = SelfiesTokenizer.build([["[C][N][=O]"]])
    p = tmp_path / "vocab.json"
    tok.save(p)
    tok2 = SelfiesTokenizer.load(p)
    assert tok2.stoi == tok.stoi
    assert len(tok2) == len(tok)

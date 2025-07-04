def assert_same_length(*sequences):
    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        raise ValueError(f"Sequences have different lengths: {[len(seq) for seq in sequences]}")
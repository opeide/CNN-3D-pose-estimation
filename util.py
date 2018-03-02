def to_triplets(arr):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(arr), 3):
        yield arr[i:i + 3]
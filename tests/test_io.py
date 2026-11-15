from pathlib import Path

import numpy as np

from numcompute.io import read_csv

# Build the path from this file's location, so the test works no matter
# which folder pytest is started from.
SAMPLE_CSV = Path(__file__).parent / "sample.csv"


def test_read_csv_basic():
    # Load a sample CSV file using our custom reader
    data = read_csv(SAMPLE_CSV, delimiter=",")

    # The output should always be a NumPy array
    assert isinstance(data, np.ndarray)

    # Basic shape check (sample.csv has 2 rows and 2 columns)
    assert data.shape == (2, 2)

    # Check a couple of values to ensure correct loading
    assert data[0, 0] == 1
    assert data[1, 1] == 4

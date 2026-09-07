"""
Unit tests for network.flow.csv_loader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from network.flow.csv_loader import CSVFlowLoader, DataValidationError, parse_timestamp_scalar


def test_parse_timestamp_scalar():
    # Epoch float
    assert parse_timestamp_scalar(1680000000.0) == 1680000000.0
    # Epoch int
    assert parse_timestamp_scalar(1680000000) == 1680000000.0
    # Millisecond epoch
    assert parse_timestamp_scalar(1680000000000) == 1680000000.0
    # Microsecond epoch
    assert parse_timestamp_scalar(1680000000000000) == 1680000000.0
    # ISO string
    ts = parse_timestamp_scalar("2023-03-28 10:00:00")
    assert isinstance(ts, float)
    assert ts > 0

    with pytest.raises(ValueError):
        parse_timestamp_scalar(np.nan)


def test_csv_loader_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    loader = CSVFlowLoader()
    df = loader.load_dataframe(empty_file)
    assert df.empty
    flows = loader.load_flows(empty_file)
    assert flows == []


def test_csv_loader_missing_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("random_col_1,random_col_2\n1,2\n")

    loader = CSVFlowLoader()
    with pytest.raises(DataValidationError) as exc:
        loader.load_dataframe(bad_csv)
    assert "missing mandatory flow columns" in str(exc.value)


def test_csv_loader_nan_and_inf_handling(tmp_path: Path):
    csv_file = tmp_path / "nans_and_infs.csv"
    csv_file.write_text(
        "timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packets,bytes,duration,syn_flag\n"
        "1680000000.0,192.168.1.1,10.0.0.1,50000,80,6,10,Infinity,0.5,1\n"
        "1680000005.0,192.168.1.2,10.0.0.1,50001,80,6,NaN,1200,-Infinity,0\n"
    )

    loader = CSVFlowLoader()
    df = loader.load_dataframe(csv_file)
    assert len(df) == 2
    assert not np.isinf(df["bytes"].to_numpy()).any()
    assert not np.isnan(df["bytes"].to_numpy()).any()
    assert not np.isnan(df["packets"].to_numpy()).any()


def test_csv_loader_sorting_and_deduplication(tmp_path: Path):
    csv_file = tmp_path / "unsorted_and_dups.csv"
    csv_file.write_text(
        "timestamp,src_ip,dst_ip,src_port,dst_port,protocol,packets,bytes\n"
        "1680000050.0,192.168.1.1,10.0.0.1,50000,80,6,10,500\n"
        "1680000010.0,192.168.1.1,10.0.0.1,50000,80,6,10,500\n"  # earlier
        "1680000010.0,192.168.1.1,10.0.0.1,50000,80,6,10,500\n"  # duplicate
    )

    loader = CSVFlowLoader()
    df = loader.load_dataframe(csv_file)
    assert len(df) == 2
    # Ensure sorted ascending
    assert df["timestamp"].iloc[0] == 1680000010.0
    assert df["timestamp"].iloc[1] == 1680000050.0


def test_csv_loader_load_flows(tmp_path: Path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "Timestamp,Source IP,Destination IP,Source Port,Destination Port,Protocol,Total Fwd Packets,Total Bytes\n"
        "1680000000.0,192.168.1.10,192.168.1.1,45000,445,6,15,900\n"
    )

    loader = CSVFlowLoader()
    flows = loader.load_flows(csv_file)
    assert len(flows) == 1
    f = flows[0]
    assert f.src_ip == "192.168.1.10"
    assert f.dst_port == 445
    assert f.packets == 15
    assert f.bytes == 900

"""
Unit tests for ml.state.state_builder.NetworkStateBuilder.
"""

import pytest
import numpy as np
import pandas as pd
from network.flow.flow_record import FlowRecord
from ml.state.state_builder import NetworkStateBuilder, FEATURE_NAMES, shannon_entropy


def test_shannon_entropy():
    # Uniform 2 classes: p1 = 0.5, p2 = 0.5 -> H = -2*(0.5 * log2(0.5)) = 1.0 bit
    assert np.isclose(shannon_entropy([10, 10]), 1.0)
    # Single class: H = 0.0
    assert shannon_entropy([50]) == 0.0
    # Empty
    assert shannon_entropy([]) == 0.0


def test_state_builder_aggregation():
    builder = NetworkStateBuilder(window_size_seconds=30.0)

    # Construct 4 flows across two 30-sec windows:
    # Window 0: [100.0, 130.0) -> flows at 105.0, 120.0
    # Window 1: [130.0, 160.0) -> flows at 135.0, 145.0
    flows = [
        FlowRecord(
            timestamp=105.0,
            src_ip="192.168.1.5",
            dst_ip="10.0.0.1",
            src_port=50000,
            dst_port=445, # SMB
            protocol=6,
            packets=10,
            bytes=1000,
            syn_flag=1,
            scenario_id="sc_1",
            label="SMB_Spread",
        ),
        FlowRecord(
            timestamp=120.0,
            src_ip="192.168.1.6",
            dst_ip="10.0.0.2",
            src_port=50001,
            dst_port=445, # SMB
            protocol=6,
            packets=20,
            bytes=2000,
            syn_flag=1,
            scenario_id="sc_1",
            label="SMB_Spread",
        ),
        FlowRecord(
            timestamp=135.0,
            src_ip="192.168.1.7",
            dst_ip="10.0.0.3",
            src_port=50002,
            dst_port=80, # Web
            protocol=6,
            packets=5,
            bytes=500,
            syn_flag=0,
            scenario_id="sc_1",
            label="BENIGN",
        ),
        FlowRecord(
            timestamp=145.0,
            src_ip="192.168.1.8",
            dst_ip="10.0.0.3",
            src_port=50003,
            dst_port=443, # Web
            protocol=6,
            packets=15,
            bytes=1500,
            syn_flag=0,
            scenario_id="sc_1",
            label="BENIGN",
        ),
    ]

    df_states = builder.build_states(flows)

    assert len(df_states) == 2
    # Verify Window 0 values
    w0 = df_states.iloc[0]
    assert w0["window_idx"] == 0
    assert w0["flow_count"] == 2
    assert w0["total_packets"] == 30
    assert w0["total_bytes"] == 3000
    assert np.isclose(w0["pkt_rate"], 30.0 / 30.0)
    assert np.isclose(w0["byte_rate"], 3000.0 / 30.0)
    assert w0["unique_src_ips"] == 2
    assert w0["unique_dst_ips"] == 2
    assert w0["unique_dst_ports"] == 1
    assert w0["syn_count"] == 2
    assert np.isclose(w0["syn_ratio"], 1.0)
    assert np.isclose(w0["port_445_share"], 1.0)
    assert w0["port_80_443_share"] == 0.0

    # Verify Window 1 values
    w1 = df_states.iloc[1]
    assert w1["window_idx"] == 1
    assert w1["flow_count"] == 2
    assert w1["total_packets"] == 20
    assert w1["total_bytes"] == 2000
    assert w1["syn_count"] == 0
    assert w1["port_445_share"] == 0.0
    assert np.isclose(w1["port_80_443_share"], 1.0)


def test_feature_schema_completeness():
    builder = NetworkStateBuilder()
    assert len(builder.feature_names) == 24
    for feat in FEATURE_NAMES:
        assert feat in builder.feature_names

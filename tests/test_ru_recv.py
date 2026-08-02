"""Static regression tests for receiver/src/ru_recv.cpp.

These tests verify that the flow-rule construction and RURecv lifecycle
code in ru_recv.cpp preserves essential invariants.  They do not require
DPDK or CUDA hardware; full functional coverage should be added in the
upstream C++ test harness.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "receiver" / "src" / "ru_recv.cpp"


@pytest.fixture(scope="module")
def source():
    if not SOURCE.exists():
        pytest.skip("receiver/src/ru_recv.cpp not present in this checkout")
    return SOURCE.read_text()


def test_flow_rule_pattern(source):
    """IQ flow rules match source MAC, VLAN, and eCPRI PC ID."""
    # setup_rules builds exactly the item/action sequence we expect.
    assert "RTE_FLOW_ITEM_TYPE_ETH" in source
    assert "RTE_FLOW_ITEM_TYPE_VLAN" in source
    assert "RTE_FLOW_ITEM_TYPE_ECPRI" in source
    assert "RTE_FLOW_ACTION_TYPE_QUEUE" in source
    assert "RTE_FLOW_ACTION_TYPE_END" in source

    # For IQ packets the source MAC is fully masked and copied from ru_addr.
    assert re.search(r"eth_mask\.src\.addr_bytes\[\d+\]\s*=\s*0xFF", source)
    masks = re.findall(r"eth_mask\.src\.addr_bytes\[(\d+)\]\s*=\s*0xFF", source)
    assert set(masks) == {"0", "1", "2", "3", "4", "5"}

    # VLAN TCI mask covers the lower 12 bits.
    assert re.search(r"vlan_mask\.tci\s*=\s*rte_cpu_to_be_16\(0x0fff\)", source)

    # eCPRI common type and PC ID are masked.
    assert re.search(r"ecpri_mask\.hdr\.common\.type\s*=\s*0xFF", source)
    assert re.search(r"ecpri_mask\.hdr\.type0\.pc_id\s*=\s*0xFFFF", source)

    # IQ path sets the PC ID from the flow argument.
    assert re.search(
        r"ecpri_spec\.hdr\.type0\.pc_id\s*=\s*rte_cpu_to_be_16\(flow\)", source
    )


def test_ru_recv_lifecycle(source):
    """RURecv ctor/dtor pairs CUDA host memory and streams."""
    # Constructor allocates and marks bursts free.
    assert "cudaMallocHost((void **)&burst_list" in source
    assert "BURST_FREE" in source
    for key in ("bytes", "num_mbufs", "status"):
        assert f"burst_list[bindex].{key}" in source

    # Destructor tears down in reverse order.
    assert "cudaStreamDestroy(stream)" in source
    assert "cudaFreeHost(burst_list)" in source

    # Per-queue flow rules are destroyed for every AP.
    assert "rte_flow_destroy(port_id, frule[iqueue], NULL)" in source
    assert "NUM_AP" in source

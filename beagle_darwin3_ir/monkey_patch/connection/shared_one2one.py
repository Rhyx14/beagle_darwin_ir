from darwin3_deployment.codegen import check_distance, PhysicalPopulation
import numpy as np
from typing import List

# Dendrite/axon types.
_RELATIVE_DENDRITE_TYPE = 4
_ABS_DENDRITE_TYPE = 0


def connect_shared_one2one(
        weight: int,
        pre_pops: List[PhysicalPopulation],
        post_pops: List[PhysicalPopulation],
        conn_table: np.ndarray | None = None,
        omit_zero_weight: bool = True) -> None:
    """Connect physical populations one by one with shared dendrites.

    All connections between the same (pre pop, post pop) pair share a single
    relative dendrite (`type=4`, `length=1`, inline weight), so only one
    dendrite row and one axon target entry are generated per pop pair.

    Args:
        weight: scalar connection weight shared by every connection.
        pre_pops: pre-synaptic neuron populations.
        post_pops: post-synaptic neuron populations.
        conn_table: connections in the form of [[pre, post], [pre, post], ...].
            Neuron ids go across all the pops in the list. `None` means an
            identity mapping.
        omit_zero_weight: omit connections with zero weight.

    Notes:
        - The total number of pre-neurons must equal the total number of
          post-neurons.
        - `weight` is a single scalar value, not a per-connection vector; a
          vector raises `TypeError`.
        - A post population is assumed to have at most 4096 neurons, so the
          relative axon base always fits the 12-bit axon field.
        - When sharing a relative dendrite is unsafe (a pre neuron connected
          more than once in a group, or an existing axon base conflict), the
          group falls back to ordinary per-connection dendrites (`type=0`).
    """
    total_pre = _populations_size(pre_pops)
    total_post = _populations_size(post_pops)
    assert total_pre == total_post, (
        f"Number of pre-neurons should equal to the post-neurons, "
        f"got 'pre {total_pre} - post {total_post}' instead")

    conn_table = _resolve_conn_table(conn_table, total_pre, total_post)

    if np.ndim(weight) != 0:
        raise TypeError(
            f"weight should be a scalar, got an array of shape {np.shape(weight)}")
    weight = int(weight)

    pre_offsets = _populations_offsets(pre_pops)
    post_offsets = _populations_offsets(post_pops)

    # (pre pop, post pop) -> [(pre local id, post local id), ...]
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    if not (omit_zero_weight and weight == 0):
        for _global_pre_id, _global_post_id in conn_table:
            _global_pre_id = int(_global_pre_id)
            _global_post_id = int(_global_post_id)
            pre_pop_id, pre_neuron_id = _locate(pre_offsets, _global_pre_id)
            post_pop_id, post_neuron_id = _locate(post_offsets, _global_post_id)
            groups.setdefault((pre_pop_id, post_pop_id), []).append(
                (pre_neuron_id, post_neuron_id))

    for (pre_pop_id, post_pop_id), pairs in groups.items():
        _add_shared_one2one_group(
            pre_pops[pre_pop_id], post_pops[post_pop_id], pairs, weight)


def _populations_size(pops: List[PhysicalPopulation]) -> int:
    return sum(_pop.size for _pop in pops)


def _populations_offsets(pops: List[PhysicalPopulation]) -> np.ndarray:
    """Offsets of the global ids of every population."""
    return np.concatenate(([0], np.cumsum([_pop.size for _pop in pops])))


def _locate(offsets: np.ndarray, global_id: int) -> tuple[int, int]:
    """Map a global neuron id to (population index, local neuron id)."""
    pop_id = int(np.searchsorted(offsets, global_id, side="right") - 1)
    return pop_id, global_id - int(offsets[pop_id])


def _resolve_conn_table(conn_table: np.ndarray | None, total_pre: int,
                        total_post: int) -> np.ndarray:
    if conn_table is None:
        return np.arange(total_pre, dtype=np.int64)[:, None].repeat(2, axis=1)

    conn_table = np.asarray(conn_table)
    assert conn_table.ndim == 2 and conn_table.shape[1] == 2, (
        f"conn_table should be in shape (N, 2), got {conn_table.shape}")
    assert conn_table.shape[0] == total_pre, (
        f"Expect {total_pre} connections, got {conn_table.shape[0]} instead")
    assert np.issubdtype(conn_table.dtype, np.integer), (
        f"conn_table should contain integer ids, got dtype {conn_table.dtype}")

    pre_ids, post_ids = conn_table[:, 0], conn_table[:, 1]
    assert np.all((pre_ids >= 0) & (pre_ids < total_pre)), (
        f"pre-neuron ids out of range [0, {total_pre})")
    assert np.all((post_ids >= 0) & (post_ids < total_post)), (
        f"post-neuron ids out of range [0, {total_post})")
    return conn_table


def _can_share_dendrite(pre_pop: PhysicalPopulation,
                        pairs: list[tuple[int, int]]) -> bool:
    """Whether the group can be encoded by one relative (`type=4`) dendrite."""
    seen = set()
    for pre_neuron_id, post_neuron_id in pairs:
        if pre_neuron_id in seen:  # one axon carries one base per target pop
            return False
        seen.add(pre_neuron_id)
        if pre_pop._axons[pre_neuron_id]:
            # `add_axon` cannot rewrite an existing, different neuron offset
            existing = pre_pop._axons[pre_neuron_id][0]
            if existing >= 0 and existing != post_neuron_id:
                return False
    return True


@check_distance
def _add_shared_one2one_group(pre_pop: PhysicalPopulation, post_pop: PhysicalPopulation,
                              pairs: list[tuple[int, int]], weight: int) -> None:
    """Add one (pre pop, post pop) group of one-to-one connections.

    Args:
        pre_pop: 突触前神经元群
        post_pop: 突触后神经元群
        pairs: [(pre local id, post local id), ...]
        weight: shared connection weight

    Notes:
        A single relative dendrite is registered for the whole group, whose
        neuron-id table row is `[0]` and whose axon base is the local post id.
        The group falls back to per-connection dendrites when sharing is unsafe.
    """
    if not _can_share_dendrite(pre_pop, pairs):
        _add_fallback_connections(pre_pop, post_pop, pairs, weight)
        return

    weight = int(weight)
    neuron_table_id = post_pop._register(
        post_pop._neuron_ids, post_pop._neuron_ids_registry, [0])
    dendrite = (neuron_table_id, _RELATIVE_DENDRITE_TYPE, 1, True, weight)
    den_id = post_pop._register(
        post_pop._dendrites, post_pop._dendrites_registry, dendrite)

    for pre_neuron_id, post_neuron_id in pairs:
        pre_pop.add_axon(pre_neuron_id, post_neuron_id, post_pop, den_id)
    return


def _add_fallback_connections(pre_pop: PhysicalPopulation, post_pop: PhysicalPopulation,
                              pairs: list[tuple[int, int]], weight: int) -> None:
    """Add the group as ordinary (`type=0`) per-connection dendrites."""
    weight = int(weight)
    for pre_neuron_id, post_neuron_id in pairs:
        den_id = post_pop.add_dendrite(
            [post_neuron_id], weight, share=True, type=_ABS_DENDRITE_TYPE)
        pre_pop.add_axon(pre_neuron_id, -1, post_pop, den_id)
    return

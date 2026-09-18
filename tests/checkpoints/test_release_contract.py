"""Published checkpoint modules preserve the public serialization contract."""

from anemone.checkpoints import (
    build_atoms,
    build_search_checkpoint_payload,
    deserialize_checkpoint_atom,
    load_search_from_checkpoint_payload,
    serialize_checkpoint_atom,
)


def test_checkpoint_atom_module_and_existing_entrypoints() -> None:
    """The extracted module and legacy public functions remain importable together."""
    branch = ("morpion", 3, -4)
    payload = build_atoms.serialize_checkpoint_atom(branch)

    assert payload == serialize_checkpoint_atom(branch)
    assert deserialize_checkpoint_atom(payload) == branch
    assert callable(build_search_checkpoint_payload)
    assert callable(load_search_from_checkpoint_payload)

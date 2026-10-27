"""Short-allocation validation preserves completed tasks and rejects mixed evidence."""

import pytest

from nnInteractive.automatic.validation_state import TaskJournal


def test_rank_partition_and_partial_write_recovery(tmp_path):
    for rank in range(2):
        journal = TaskJournal(tmp_path, {"weight": 1}, rank, 2, max_tasks=1)
        for index in journal.indices(5):
            journal.record(index, [{"index": index}])
        assert len(journal.tasks) == 1
        with journal.path.open("ab") as handle:
            handle.write(b'{"task_index":')
        resumed = TaskJournal(tmp_path, {"weight": 1}, rank, 2)
        for index in resumed.indices(5):
            resumed.record(index, [{"index": index}])
        assert resumed.complete(5)
        assert resumed.rows() == [{"index": i} for i in range(rank, 5, 2)]
        with pytest.raises(ValueError, match="duplicate"):
            resumed.record(rank, [{"index": rank}])
    with pytest.raises(ValueError, match="mismatch"):
        TaskJournal(tmp_path, {"weight": 1}, 0, 1)

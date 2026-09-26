# SPDX-License-Identifier: Apache-2.0
"""Restore per-record context for Inspector v4.3, accepting older JSON too."""


class DumpContext:
    """One instance per input file; never carry context between files."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.header = None
        self.metadata = None
        self.remaining = {}

    def restore(self, record):
        if not isinstance(record, dict):
            raise ValueError("Inspector record must be an object")
        if "dump_stats" in record:
            if any(self.remaining.values()):
                raise ValueError("Incomplete or interleaved Inspector dump")
            self.reset()
            if record.get("metadata", {}).get("inspector_output_format_version") == "v4.3":
                self.header = record["header"]
                self.metadata = record["metadata"]
                stats = record["dump_stats"]
                self.remaining = {"coll_perf": stats["coll_records"],
                                  "p2p_perf": stats["p2p_records"],
                                  "proxy_trace": stats.get("proxy_records", 0)}
                if any(type(n) is not int or n < 0 for n in self.remaining.values()):
                    raise ValueError("Invalid Inspector dump counts")
            return record
        if self.header is not None:
            if len(record) != 1:
                raise ValueError("Unexpected fields in Inspector v4.3 payload")
            kind = next(iter(record))
            if self.remaining.get(kind, 0) <= 0:
                raise ValueError("Unexpected record beyond Inspector dump counts")
            self.remaining[kind] -= 1
            return {"header": self.header, "metadata": self.metadata, **record}
        # v4.0-v4.2 records carry their own context. An orphan v4.3 payload
        # must not silently acquire an unknown communicator or the previous PID.
        if "header" not in record or "metadata" not in record:
            raise ValueError("Missing Inspector record context")
        return record

    def finish(self):
        if any(self.remaining.values()):
            raise ValueError("Truncated Inspector dump")

#!/usr/bin/env python3
"""Hadoop Streaming REDUCER (also usable as COMBINER).

Input : ``<metric>|<dimension>\\t<count>`` lines, sorted by key (Hadoop's shuffle/sort).
Output: ``<metric>|<dimension>\\t<total>`` - same format, so it can run as a combiner.
"""
import sys


def main():
    current_key = None
    total = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        key, sep, value = line.rpartition("\t")
        if not sep:
            continue
        try:
            count = int(value)
        except ValueError:
            sys.stderr.write("reporter:counter:SafetyAnalytics,bad_reducer_values,1\n")
            continue
        if key == current_key:
            total += count
        else:
            if current_key is not None:
                sys.stdout.write("%s\t%d\n" % (current_key, total))
            current_key, total = key, count
    if current_key is not None:
        sys.stdout.write("%s\t%d\n" % (current_key, total))


if __name__ == "__main__":
    main()

#!/usr/bin/python
# -*- coding: utf-8 -*-

import typing

from collections import namedtuple
from functools import lru_cache

Item = namedtuple("Item", ['index', 'value', 'weight'])
EItem = namedtuple("EItem", ["index", "profit", "weight", "value"])
Node = namedtuple("Node",["value","room","estimate","parent","next_item","choosen","unchoosen"])

# Algorithms -- Start
def greedy(items: typing.List[Item], capacity: int):
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in sorted(items, key=lambda x: x.value, reverse=True):
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


# Branch and bound -- Start
@lru_cache(maxsize=None)
def get_relaxed_estimation(items, capacity):
    stack = sorted(
                [EItem(i.index, i.value/i.weight, i.weight, i.value) for i in items],
                key=lambda x: x.profit)
    weight = 0
    estimation = 0
    while stack and weight < capacity:
        item = stack.pop()
        if weight + item.weight <= capacity:
            weight += item.weight
            estimation += item.value
        else:
            estimation += item.profit * (capacity - weight)
            weight = capacity
    return estimation

def prepare_root(items: typing.List[Item], capacity: int) -> Node:
    return Node(
        value=0,
        room=capacity,
        estimate=sum(i.value for i in items),
        parent=None, next_item=items[-1],
        choosen=set(), unchoosen=set())


def branch_and_bound(items, capacity):
    value = 0
    taken = [0]*len(items)

    items = sorted(items, key=lambda x: x.value/x.weight)
    items_set = set(items)

    root = prepare_root(items, capacity)
    max_node = root
    stack = [root]

    while stack:
        prev_node = stack.pop()
        cur_item = prev_node.next_item
        cur_pos = items.index(cur_item)
        next_item = items[cur_pos - 1] if cur_pos != 0 else None

        left, right = None, None
        # x = 0
        right_unchoosen = prev_node.unchoosen.copy()
        right_unchoosen.add(cur_item)
        right_estimate = prev_node.value + get_relaxed_estimation(frozenset(
                                    items_set - right_unchoosen - prev_node.choosen.copy()),
                                    prev_node.room)
        # nodes.append(right)
        if (next_item != None and
            right_estimate >= max_node.value
        ):
            right = Node(
                value=prev_node.value,
                room=prev_node.room,
                estimate=right_estimate,
                parent=prev_node,
                next_item=next_item,
                choosen=prev_node.choosen.copy(),
                unchoosen=right_unchoosen)
            stack.append(right)

        # x = 1
        left_room = prev_node.room - cur_item.weight
        if left_room >= 0:

            left_choosen = prev_node.choosen.copy()
            left_choosen.add(cur_item)

            left = Node(
                value=prev_node.value + cur_item.value,
                room=left_room,
                estimate=prev_node.estimate,
                parent=prev_node,
                next_item=next_item,
                choosen=left_choosen,
                unchoosen=prev_node.unchoosen.copy())
            if (left.next_item != None and
                left.estimate >= max_node.value):
                stack.append(left)
            if max_node.value < left.value:
                max_node = left

    value = max_node.value
    for item in max_node.choosen:
        taken[item.index] = 1

    return value, taken
# Branch and bound -- End
# Algorithms -- End

def read_file(input_data):
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    return capacity, items


def solve_it(input_data, method="opt"):
    # Modify this code to run your optimization algorithm

    # parse the input
    capacity, items = read_file(input_data)

    if method == "opt":
        value, taken = branch_and_bound(items, capacity)
    else:
        value, taken = greedy(items, capacity)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))

    assert sum(items[index].value if value == 1 else 0 for index, value in enumerate(taken)) == value
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

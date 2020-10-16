import random
import math
import json
from copy import deepcopy


class Tran:
    def __init__(self, tran_id, source, action, guards, reset, target):
        self.tran_id = tran_id
        self.source = source
        self.action = action
        self.guards = guards
        self.reset = reset
        self.target = target

    def show(self):
        return {self.tran_id: [self.source, self.action, self.guards, self.reset, self.target]}


def main():
    # 配置参数
    file_name = '4_2_10'
    state_num, input_num, guard_upper_bound = [int(i) for i in file_name.split('_')]
    partition_size = 2

    # 开始生成
    name = file_name
    inputs = get_inputs(input_num)
    states = get_states(state_num)
    initState = '0'
    acceptStates = get_acceptStates(states)
    trans = get_trans(inputs, states, guard_upper_bound, partition_size)
    while not validate_trans(trans, inputs):  # 需要包含所有inputs
        trans = get_trans(inputs, states, guard_upper_bound, partition_size)

    # 生成json文件
    build_json_file('model', name, inputs, states, initState, acceptStates, trans)
    return True


def get_inputs(input_num):
    return list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")[:input_num]


def get_states(state_num):
    return [str(i) for i in range(state_num)]


def get_acceptStates(states):
    accept_num = random.randint(int(math.ceil(len(states) / 8)), int(math.ceil(len(states) / 2)))
    acceptStates = random.sample(states, accept_num)
    acceptStates.sort()
    return acceptStates


def get_trans(inputs, states, guard_upper_bound, partition_size):
    traveled = []
    trans = []
    while len(traveled) != len(states):  # 如果一轮后未所有状态都访问到则无效，重新开始
        untraveled = set('0')
        tran_id = 0
        traveled = []
        trans = []
        while len(untraveled) > 0:
            source = untraveled.pop()
            temp_trans, reach_states, tran_id = get_trans_from_curState(source, tran_id, states, inputs, guard_upper_bound, partition_size)
            trans += temp_trans
            if len(temp_trans) > 0:
                traveled.append(source)
            for state in reach_states:
                if state not in traveled:
                    untraveled.add(state)
    return trans


def get_trans_from_curState(source, tran_id, states, inputs, guard_upper_bound, partition_size):
    cur_tran_id = tran_id
    trans = []
    reach_states = set()
    for input in inputs:
        if random.random() > 0.5:
            continue
        next_trans_num, guards = get_random_guards(guard_upper_bound, partition_size)
        for i in range(next_trans_num):
            target = random.choice(states)
            reset = random.choice(['r', 'r', 'r', 'n'])
            temp_tran = Tran(str(cur_tran_id), source, input, guards[i], reset, target)
            trans.append(temp_tran)
            cur_tran_id += 1
            if target not in reach_states:
                reach_states.add(target)
    return trans, reach_states, cur_tran_id


def get_random_guards(guard_upper_bound, partition_size):
    guards = []
    endpoint_set = {0}
    while len(endpoint_set) < partition_size:
        endpoint = random.randint(0, guard_upper_bound - 1)
        endpoint_set.add(endpoint)
    endpoint_list = list(endpoint_set)
    endpoint_list.sort()
    endpoint_list.append('+')
    right = None
    for i in range(len(endpoint_list) - 1):
        if right is None:
            left = random.choice(['[', '('])
        elif right == ']':
            left = '('
        else:
            left = '['
        right = random.choice([']', ')'])
        if i == len(endpoint_list) - 2:
            right = ')'
        temp_guard = left + str(endpoint_list[i]) + ',' + str(endpoint_list[i + 1]) + right
        guards.append(temp_guard)
    guards_num = random.randint(1, partition_size)
    guards = random.sample(guards, guards_num)
    return guards_num, guards


def validate_trans(trans, inputs):
    temp_inputs = deepcopy(inputs)
    for tran in trans:
        if tran.action in temp_inputs:
            temp_inputs.remove(tran.action)
    return len(temp_inputs) == 0


def build_json_file(file_name, name, inputs, states, initState, acceptStates, trans):
    tran_dict = {}
    for t in trans:
        t_dict = t.show()
        tran_dict[tuple(t_dict.keys())[0]] = tuple(t_dict.values())[0]
    model_dict = {
        "name": name,
        "states": states,
        "inputs": inputs,
        "trans": tran_dict,
        "initState": initState,
        "acceptStates": acceptStates
    }
    text = json.dumps(model_dict)
    with open(file_name + '.json', 'w') as f:
        f.write(json_format(text))


def json_format(text):
    text_list = []
    left_brace_num = 0
    bracket_num = 0
    for i in range(len(text)):
        text_list.append(text[i])
        if text[i] == '[' and left_brace_num == 1:
            bracket_num = bracket_num + 1
        if text[i] == ']' and left_brace_num == 1:
            bracket_num = bracket_num - 1
        if text[i] == '{':
            left_brace_num = left_brace_num + 1
            if left_brace_num == 1:
                text_list.append('\n' + " " * 4)
            else:
                bracket_num = 0
                text_list.append('\n' + " " * 4 + "    " * (left_brace_num - 1))
        if text[i] == '}':
            left_brace_num = left_brace_num - 1
        if text[i] == ',':
            if text[i - 1] == ']' and left_brace_num == 2:
                if left_brace_num == 2:
                    text_list.append('\n' + " " * 7)
                if left_brace_num == 1:
                    text_list.append('\n' + " " * 3)
            elif text[i - 1] == '}':
                text_list.append('\n' + " " * 3)
                bracket_num = 0
            elif left_brace_num == 1 and bracket_num == 0:
                text_list.append('\n' + " " * 3)
        if text[i] == ']':
            if text[i + 1] == '}' and left_brace_num == 2:
                text_list.append('\n' + " " * 4)
            if text[i + 1] == '}' and left_brace_num == 1:
                text_list.append('\n')
    format_str = "".join(text_list)
    return format_str


if __name__ == '__main__':
    main()

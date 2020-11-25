# 随机测试算法
import random
import copy
import queue
import math
from common.TimedWord import TimedWord


# 随机测试算法1 - 完全随机采样
def random_testing_1(hypothesis, upper_guard, state_num, system):
    test_num = int(len(hypothesis.states) * len(hypothesis.actions) * upper_guard * 10)

    ctx = None
    for i in range(test_num):
        test = test_generation_1(hypothesis.actions, upper_guard, state_num)
        flag = test_execution(hypothesis, system, test)
        if flag:
            ctx = test
            break
    if ctx is not None:
        return False, ctx
    return True, ctx


# 测试集生成方法（随机测试算法1）
def test_generation_1(actions, upper_guard, state_num):
    test = []
    length = random.randint(1, state_num * 2)
    for i in range(length):
        action = actions[random.randint(0, len(actions) - 1)]
        time = get_random_delay(upper_guard)
        temp = TimedWord(action, time)
        test.append(temp)
    return test


# 随机测试算法2 - 源自：Efficient Active Automata Learning via Mutation Testing
def random_testing_2(hypothesis, upper_guard, state_num, system):
    test_num = int(len(hypothesis.states) * len(hypothesis.actions) * upper_guard * 10)
    pretry = 0.9
    pstop = 0.1
    linfix = min(math.ceil(len(hypothesis.states) / 2), math.ceil(state_num / 2))
    max_steps = int(1.5 * state_num)

    ctx = None
    for i in range(test_num):
        test = test_generation_2(hypothesis, pretry, pstop, max_steps, linfix, upper_guard)
        test_list = prefixes(test)
        for j in test_list:
            flag = test_execution(hypothesis, system, j)
            if flag:
                ctx = test
                return False, ctx
    return True, ctx


# 测试集生成方法（随机测试算法2）
def test_generation_2(hypothesis, pretry, pstop, max_steps, linfix, upper_guard):
    test = []
    hypothesis = copy.deepcopy(hypothesis)
    li = random.randint(1, linfix)
    now_time = 0
    state = hypothesis.init_state
    if coin_flip(0.5):
        actions = []
        for i in range(li):
            actions.append(random.choice(hypothesis.actions))
        for action in actions:
            time = get_random_delay(upper_guard)
            temp_DTW = TimedWord(action, time)
            temp_LTW = TimedWord(action, now_time + time)
            test.append(temp_DTW)
            for t in hypothesis.trans:
                if t.source == state and t.is_passing_tran(temp_LTW):
                    state = t.target
                    if t.reset:
                        now_time = 0
                    else:
                        now_time = temp_LTW.time
                    break
    while True:
        rS = random.choice(hypothesis.states)
        p0, now_time = find_path(hypothesis, upper_guard, now_time, state, rS)
        if p0:
            li = random.randint(1, linfix)
            rSteps_i = []
            for i in range(li):
                rSteps_i.append(random.choice(hypothesis.actions))
            rSteps = []
            for rsi in rSteps_i:
                time = get_random_delay(upper_guard)
                rsi_temp_DTW = TimedWord(rsi, time)
                rsi_temp_LTW = TimedWord(rsi, now_time + time)
                rSteps.append(rsi_temp_DTW)
                for t in hypothesis.trans:
                    if t.source == state and t.is_passing_tran(rsi_temp_LTW):
                        state = t.target
                        if t.reset:
                            now_time = 0
                        else:
                            now_time = rsi_temp_LTW.time
                        break
            test = test + p0 + rSteps
            if len(test) > max_steps:
                break
            elif coin_flip(pstop):
                break
        elif coin_flip(1 - pretry):
            break
    return test


# 随机测试算法3 - 源自：Active Model Learning of Timed Automata via Genetic Programming
def random_testing_3(hypothesis, upper_guard, state_num, system):
    test_num = int(len(hypothesis.states) * len(hypothesis.actions) * upper_guard * 10)
    n_len = int(state_num * 2)
    p_valid = 0.9
    p_delay = 0.6

    ctx = None
    for i in range(test_num):
        test = test_generation_3(hypothesis, n_len, p_valid, p_delay, upper_guard)
        test_list = prefixes(test)
        for j in test_list:
            flag = test_execution(hypothesis, system, j)
            if flag:
                ctx = test
                return False, ctx
    return True, ctx


# 测试集生成方法（随机测试算法3）
def test_generation_3(hypothesis, n_len, p_valid, p_delay, upper_guard):
    test = []
    state_now = hypothesis.init_state
    time_now = 0
    while len(test) < n_len:
        transition = None
        delay = get_random_delay(upper_guard)
        transition_list = get_transition_list(hypothesis, state_now)
        if random.random() <= p_valid:
            if random.random() <= p_delay:
                delay_list = get_delay_list(transition_list)
                if len(delay_list) > 0:
                    delay = random.choice(delay_list)
            cur_time = time_now + delay
            transition_valid_list = []
            for tran in transition_list:
                for guard in tran.guards:
                    if guard.is_in_interval(cur_time):
                        if tran.target != tran.source or tran.reset:
                            transition_valid_list.append(tran)
                            break
            if len(transition_valid_list) > 0:
                transition = random.choice(transition_valid_list)

        if transition is None:
            cur_time = time_now + delay
            transition_invalid_list = []
            for tran in transition_list:
                if tran.target == tran.source and not tran.reset:
                    for guard in tran.guards:
                        if guard.is_in_interval(cur_time):
                            transition_invalid_list.append(tran)
                            break
            if len(transition_invalid_list) > 0:
                transition = random.choice(transition_invalid_list)

        if transition is not None:
            test.append(TimedWord(transition.action, delay))
            if transition.reset:
                time_now = 0
            else:
                time_now = time_now + delay
            state_now = transition.target
    return test


# 随机测试算法4 - 改自随机测试算法2
def random_testing_4(hypothesis, upper_guard, state_num, system, prectxs):
    test_num = int(len(hypothesis.states) * len(hypothesis.actions) * upper_guard * 5)
    pretry = 0.9
    pstop = 0.05
    pvalid = 0.9
    pnext = 0.95
    max_steps = min(int(2 * state_num), int(2 * len(hypothesis.states)))

    for c in prectxs:
        if len(c) >= max_steps:
            prectxs.remove(c)

    ctx = None
    for i in range(test_num):
        if prectxs:
            prectx = random.choice(prectxs)
        else:
            prectx = []
        #test = test_generation_4(hypothesis, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx)
        test = test_generation_5(hypothesis, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx)
        test_list = prefixes(test)
        for j in test_list:
            flag = test_execution(hypothesis, system, j)
            if flag:
                ctx = test
                return False, ctx
    return True, ctx


# 测试集生成方法
def test_generation_4(hypothesis, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx):
    test = []
    hypothesis = copy.deepcopy(hypothesis)
    # 将迁移按照状态/有效性进行分组
    invalid_tran_dict = {}
    valid_tran_dict = {}
    for state in hypothesis.states:
        invalid_tran_dict[state] = []
        valid_tran_dict[state] = []
    for tran in hypothesis.trans:
        if tran.source == hypothesis.sink_state or tran.target == hypothesis.sink_state:
            invalid_tran_dict[tran.source].append(tran)
        else:
            valid_tran_dict[tran.source].append(tran)
    # 开始随机游走
    now_time = 0
    state = hypothesis.init_state

    if prectx and coin_flip(0.5):
        for t in prectx:
            #temp_time = now_time + t.time
            temp_LTW = TimedWord(t.action, now_time + t.time)
            for tran in valid_tran_dict[state]:
                if tran.is_passing_tran(temp_LTW):
                    state = tran.target
                    if tran.reset:
                        now_time = temp_LTW.time
                    else:
                        now_time = 0

    # 开始随机游走
    while True:
        if coin_flip(pvalid):
            if valid_tran_dict[state]:
                next_tran = random.choice(valid_tran_dict[state])
                delay_time = get_time_from_tran(next_tran, now_time, upper_guard)
                if delay_time is None:
                    continue
                test.append(TimedWord(next_tran.action, delay_time))
                state = next_tran.target
                if next_tran.reset:
                    now_time = 0
                else:
                    now_time += delay_time
            else:
                continue
        else:
            if invalid_tran_dict[state]:
                next_tran = random.choice(invalid_tran_dict[state])
                delay_time = get_time_from_tran(next_tran, now_time, upper_guard)
                if delay_time is None:
                    continue
                test.append(TimedWord(next_tran.action, delay_time))
                state = next_tran.target
                if next_tran.reset:
                    now_time = 0
                else:
                    now_time += delay_time
            else:
                continue
        if state == hypothesis.sink_state:
            break
        if len(test) > max_steps:
            break
        elif coin_flip(pstop):
            break
    # 是否多走几步，如果为sink_state则随机走几步

    if state == hypothesis.sink_state:
        linfix = math.ceil(len(hypothesis.states) / 2)
        li = random.randint(1, linfix)
        for i in range(li):
            test.append(TimedWord(random.choice(hypothesis.actions), get_random_delay(upper_guard)))
            if len(test) > max_steps:
                break
    elif coin_flip(pnext):
        target_state = random.choice(hypothesis.states)
        while True:
            path_dtw, now_time = find_path(hypothesis, upper_guard, now_time, state, target_state)
            if path_dtw:
                test.extend(path_dtw)
                break
            elif coin_flip(1 - pretry):
                break
    '''
    if coin_flip(pnext):
        target_state = random.choice(hypothesis.states)
        while True:
            path_dtw, now_time = find_path(hypothesis, upper_guard, now_time, state, target_state)
            if path_dtw:
                test.extend(path_dtw)
                break
            elif coin_flip(1 - pretry):
                break
    '''
    return test

# 测试集生成方法
def test_generation_5(hypothesis, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx):
    test = []
    hypothesis = copy.deepcopy(hypothesis)
    # 将迁移按照状态/有效性进行分组
    invalid_tran_dict = {}
    valid_tran_dict = {}
    for state in hypothesis.states:
        invalid_tran_dict[state] = []
        valid_tran_dict[state] = []
    for tran in hypothesis.trans:
        if tran.source == hypothesis.sink_state or tran.target == hypothesis.sink_state:
            invalid_tran_dict[tran.source].append(tran)
        else:
            valid_tran_dict[tran.source].append(tran)
    # 开始随机游走
    now_time = 0
    state = hypothesis.init_state

    #是否延续ctx
    if prectx and coin_flip(0.5):
        for t in prectx:
            #temp_time = now_time + t.time
            temp_LTW = TimedWord(t.action, now_time + t.time)
            for tran in valid_tran_dict[state]:
                if tran.is_passing_tran(temp_LTW):
                    state = tran.target
                    if tran.reset:
                        now_time = temp_LTW.time
                    else:
                        now_time = 0

    # 开始随机游走
    while True:
        delay_time = get_random_delay(upper_guard)
        now_time += delay_time
        if coin_flip(pvalid):
            tran_dict = valid_tran_dict[state]
            #continue
        else:
            tran_dict = invalid_tran_dict[state]
            #continue
        Es = []
        for tran in tran_dict:
            for guard in tran.guards:
                if guard.is_in_interval(now_time):
                    Es.append(tran)
                    break
        if Es:
            selec_tran = random.choice(Es)
            state = selec_tran.target
            if selec_tran.reset:
                now_time = 0
            test.append(TimedWord(selec_tran.action, delay_time))

        if state == hypothesis.sink_state:
            break
        #if len(test) > max_steps:
        #    break
        elif coin_flip(pstop):
            break
    # 是否多走几步，如果为sink_state则随机走几步
    '''
    if coin_flip(pnext):
        linfix = math.ceil(len(hypothesis.states)/2)
        li = random.randint(1, linfix)
        for i in range(li):
            test.append(TimedWord(random.choice(hypothesis.actions), get_random_delay(upper_guard)))
            if len(test) > max_steps:
                break
    '''
    if state == hypothesis.sink_state:
        linfix = math.ceil(len(hypothesis.states) / 2)
        li = random.randint(1, linfix)
        for i in range(li):
            test.append(TimedWord(random.choice(hypothesis.actions), get_random_delay(upper_guard)))
            if len(test) > max_steps:
                break
    elif coin_flip(pnext):
        target_state = random.choice(hypothesis.states)
        while True:
            path_dtw, now_time = find_path(hypothesis, upper_guard, now_time, state, target_state)
            if path_dtw:
                test.extend(path_dtw)
                break
            elif coin_flip(1 - pretry):
                break
    return test

def test_generation_6(hypothesis, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx):
    test = []
    hypothesis = copy.deepcopy(hypothesis)
    # 将迁移按照状态/有效性进行分组
    invalid_tran_dict = {}
    valid_tran_dict = {}
    for state in hypothesis.states:
        invalid_tran_dict[state] = []
        valid_tran_dict[state] = []
    for tran in hypothesis.trans:
        if tran.source == hypothesis.sink_state or tran.target == hypothesis.sink_state:
            invalid_tran_dict[tran.source].append(tran)
        else:
            valid_tran_dict[tran.source].append(tran)
    # 开始随机游走
    now_time = 0
    state = hypothesis.init_state

    if prectx and coin_flip(0.5):
        for t in prectx:
            #temp_time = now_time + t.time
            temp_LTW = TimedWord(t.action, now_time + t.time)
            for tran in valid_tran_dict[state]:
                if tran.is_passing_tran(temp_LTW):
                    state = tran.target
                    if tran.reset:
                        now_time = temp_LTW.time
                    else:
                        now_time = 0

    # 开始随机游走
    while True:
        if coin_flip(pvalid):
            if valid_tran_dict[state]:
                next_tran = random.choice(valid_tran_dict[state])
                delay_time = get_time_from_tran(next_tran, now_time, upper_guard)
                if delay_time is None:
                    continue
                test.append(TimedWord(next_tran.action, delay_time))
                state = next_tran.target
                if next_tran.reset:
                    now_time = 0
                else:
                    now_time += delay_time
            else:
                continue
        else:
            if invalid_tran_dict[state]:
                next_tran = random.choice(invalid_tran_dict[state])
                delay_time = get_time_from_tran(next_tran, now_time, upper_guard)
                if delay_time is None:
                    continue
                test.append(TimedWord(next_tran.action, delay_time))
                state = next_tran.target
                if next_tran.reset:
                    now_time = 0
                else:
                    now_time += delay_time
            else:
                continue
        #if state == hypothesis.sink_state:
        #    break
        if len(test) > max_steps:
            break
        elif coin_flip(pstop):
            break
    # 是否多走几步，如果为sink_state则随机走几步
    if coin_flip(pnext):
        linfix = math.ceil(len(hypothesis.states) / 2)
        li = random.randint(1, linfix)
        for i in range(li):
            test.append(TimedWord(random.choice(hypothesis.actions), get_random_delay(upper_guard)))
            if len(test) > max_steps:
                break

    return test


# 测试执行
def test_execution(hypothesis, system, sample):
    system_res, real_value = system.test_DTWs(sample)
    hypothesis_res, value = hypothesis.test_DTWs(sample)
    return real_value != value


# --------------------------------- auxiliary function ---------------------------------

def coin_flip(p):
    return random.random() <= p


def get_random_delay(upper_guard):
    time = random.randint(0, upper_guard * 3 + 1)
    if time % 2 == 0:
        time = time // 2
    else:
        time = time // 2 + 0.5
    return time


# prefix set of tws （tws前缀集）
def prefixes(tws):
    new_prefixes = []
    for i in range(1, len(tws) + 1):
        temp_tws = tws[:i]
        new_prefixes.append(temp_tws)
    return new_prefixes


# find a path from s1 to s2
def find_path(hypothesis, upper_guard, now_time, s1, s2):
    init_now_time = now_time
    visited = []
    next_to_explore = queue.Queue()
    next_to_explore.put([s1, []])

    while not next_to_explore.empty():
        [sc, path] = next_to_explore.get()
        if path is None:
            path = []
        if sc not in visited:
            visited.append(sc)
            for i in hypothesis.actions:
                time = get_random_delay(upper_guard)
                temp_DTW = TimedWord(i, time)
                temp_LTW = TimedWord(i, time + now_time)
                sn = None
                for ts in hypothesis.trans:
                    if ts.source == sc and ts.is_passing_tran(temp_LTW):
                        sn = ts.target
                        if ts.reset:
                            now_time = 0
                        else:
                            now_time = temp_LTW.time
                        break
                if sn == s2:
                    path.append(temp_DTW)
                    return path, now_time
                next_to_explore.put([sn, copy.deepcopy(path).append(temp_DTW)])
    return None, init_now_time


def get_transition_list(hypothesis, state_now):
    transition_list = []
    for tran in hypothesis.trans:
        if tran.source == state_now:
            transition_list.append(tran)
    return transition_list


def get_delay_list(transition_list):
    delay_list = []
    for transition in transition_list:
        for guard in transition.guards:
            left = guard.guard.split(',')[0]
            right = guard.guard.split(',')[1]
            if left[0] == '(':
                delay_list.append(guard.get_min() + 0.5)
            else:
                delay_list.append(guard.get_min())
            if right[-1] == ']':
                delay_list.append(guard.get_max())
            else:
                if right[0] != '+':
                    delay_list.append(guard.get_max() - 0.5)
    return delay_list


def get_time_from_tran(tran, now_time, upper_guard):
    valid_guards = []
    for guard in tran.guards:
        if guard.get_max() < now_time:
            continue
        elif guard.get_max() == now_time and not guard.get_closed_max():
            continue
        valid_guards.append(guard)
    if not valid_guards:
        return None
    guard = random.choice(valid_guards)

    cur_min = guard.get_min() if guard.closed_min else guard.get_min() + 0.5
    if cur_min < now_time:
        cur_min = now_time
    if guard.get_max() == float("inf"):
        return random.randint(0, upper_guard * 2) / 2
    else:
        cur_max = guard.get_max() if guard.get_closed_max() else guard.get_max() - 0.5
        time = random.randint(cur_min * 2, cur_max * 2)
        return time / 2 - now_time

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
    pstop = 0.02
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
    hypothesis = copy.deepcopy(hypothesis)
    test = []
    state = hypothesis.init_state
    li = random.randint(1, linfix)
    now_time = 0
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
        rI = random.choice(hypothesis.actions)
        time = get_random_delay(upper_guard)
        rI_temp_DTW = TimedWord(rI, time)
        rI_temp_LTW = TimedWord(rI, now_time + time)
        for t in hypothesis.trans:
            if t.source == rS and t.is_passing_tran(rI_temp_LTW):
                state = t.target
                if t.reset:
                    now_time = 0
                else:
                    now_time = rI_temp_LTW.time
                break

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
            test = test + p0 + [rI_temp_DTW] + rSteps
            if len(test) > max_steps:
                break
            elif coin_flip(pstop):
                break
        elif coin_flip(1 - pretry):
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


def find_path(hypothesis, upper_guard, now_time, s1, s2):
    # find a path from s1 to s2
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

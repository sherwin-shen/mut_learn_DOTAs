# 随机测试算法
import random
import copy
import queue
from common.TimedWord import TimedWord, DRTW_to_LRTW, LRTW_to_LTW


# 随机测试算法1 - 完全随机采样
def random_testing_1(hypothesis, upper_guard, state_num, system):
    test_num = 3000

    ctx = None
    for i in range(test_num):
        test = test_generation_1(hypothesis.actions, upper_guard, state_num)
        flag = test_execution(hypothesis, system, test)
        if flag:
            ctx = test
            break
    if ctx is not None:
        ctx = minimize_counterexample(hypothesis, system, ctx)
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
    test_num = 5000
    pretry = 0.9
    pstop = 0.02
    linfix = int(state_num / 2)
    max_steps = int(1.5 * state_num)

    ctx = None
    for i in range(test_num):
        test = test_generation_2(hypothesis, pretry, pstop, max_steps, linfix, upper_guard)
        flag = test_execution(hypothesis, system, test)
        if flag:
            ctx = test
            break
    if ctx is not None:
        ctx = minimize_counterexample(hypothesis, system, ctx)
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
            actions.append(random.sample(hypothesis.actions, 1)[0])
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
                rSteps_i.append(random.sample(hypothesis.actions, 1)[0])
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


# 随机测试算法3 - 源自：Active Model Learning of Timed Automata via Genetic Programming
def random_testing_3(hypothesis, upper_guard, state_num, system):
    test_num = 5000
    n_len = state_num * 2
    p_valid = 0.99
    p_delay = 0.6

    ctx = None
    for i in range(test_num):
        test = test_generation_3(hypothesis, n_len, p_valid, p_delay, upper_guard)
        flag = test_execution(hypothesis, system, test)
        if flag:
            ctx = test
            break
    if ctx is not None:
        ctx = minimize_counterexample(hypothesis, system, ctx)
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
            transition_valid_list = []
            for tran in transition_list:
                if tran.target == tran.source and not tran.reset:
                    for guard in tran.guards:
                        if guard.is_in_interval(cur_time):
                            transition_valid_list.append(tran)
                            break
            if len(transition_valid_list) > 0:
                transition = random.choice(transition_valid_list)
        if transition is not None:
            test.append(TimedWord(transition.action, delay))
            if transition.reset:
                time_now = 0
            else:
                time_now = time_now + delay
            state_now = transition.target
    return test


# 测试执行
def test_execution(hypothesis, system, sample):
    system_res, real_value = system.test_DTWs(sample)
    hypothesis_res, value = hypothesis.test_DTWs(sample)
    return real_value != value


# 最小化反例
def minimize_counterexample(hypothesis, system, ctx):
    ### 最小化反例的长度
    mini_ctx = []
    for dtw in ctx:
        mini_ctx.append(dtw)
        if test_execution(hypothesis, system, mini_ctx):
            break
    ### 局部最小化反例的时间
    # Find sequence of reset information
    reset = []
    DRTWs, value = system.test_DTWs(mini_ctx)
    for drtw in DRTWs:
        reset.append(drtw.reset)
    # ctx to LTWs
    LTWs = LRTW_to_LTW(DRTW_to_LRTW(DRTWs))
    # start minimize
    for i in range(len(LTWs)):
        while True:
            if i == 0 or reset[i - 1]:
                can_reduce = (LTWs[i].time > 0)
            else:
                can_reduce = (LTWs[i].time > LTWs[i - 1].time)
            if not can_reduce:
                break
            LTWs_temp = copy.deepcopy(LTWs)
            LTWs_temp[i] = TimedWord(LTWs[i].action, one_lower(LTWs[i].time))
            if not test_execution(hypothesis, system, LTW_to_DTW(LTWs_temp, reset)):
                break
            LTWs = copy.deepcopy(LTWs_temp)
        return LTW_to_DTW(LTWs, reset)


# --------------------------------- auxiliary function ---------------------------------

def one_lower(x):
    if x - int(x) == 0.5:
        return int(x)
    else:
        return x - 0.5


def LTW_to_DTW(LTWs, reset):
    DTWs = []
    for j in range(len(LTWs)):
        if j == 0 or reset[j - 1]:
            DTWs.append(TimedWord(LTWs[j].action, LTWs[j].time))
        else:
            DTWs.append(TimedWord(LTWs[j].action, LTWs[j].time - LTWs[j - 1].time))
    return DTWs


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

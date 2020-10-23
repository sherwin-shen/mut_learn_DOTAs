import math
import random
from common.TimedWord import TimedWord


def pac_testing(hypothesis, upper_guard, state_num, system):
    delta = 0.001
    epsilon = 0.001
    eq_num = system.eq_num
    test_num = int((math.log(1 / delta) + math.log(2) * (eq_num + 1)) / epsilon)

    ctx = None
    for length in range(1, math.ceil(state_num * 1.5)):  # 如果没有state_num，则自定义长度范围
        i = 0
        while i < test_num // state_num:
            i += 1
            sample = sample_generation_valid(upper_guard, length, system)
            while sample is None:
                sample = sample_generation_valid(upper_guard, length, system)
            flag = test_execution(hypothesis, system, sample)
            if flag:
                ctx = sample
                break
        if ctx is not None:
            return False, ctx
    return True, ctx


# 采样 - 有效的测试序列
def sample_generation_valid(upper_guard, length, system):
    # First produce a path (as a list of transitions) in the OTA
    path = []
    cur_state = system.init_state
    for i in range(length):
        edges = []
        for tran in system.trans:
            if cur_state == tran.source:
                edges.append(tran)
        edge = random.choice(edges)
        path.append(edge)
        cur_state = edge.target

    # Next, figure out (double of) the minimum and maximum logical time of each edge in path.
    min_time, max_time = [], []
    for tran in path:
        min_time.append(min_constraint_double(tran.guards[0]))
        max_time.append(max_constraint_double(tran.guards[0], upper_guard))

    # For each transition, maintain a mapping from logical time to the number of choices.
    weight = dict()
    for i in reversed(range(length)):
        tran = path[i]
        min_value, max_value = min_time[i], max_time[i]
        weight[i] = dict()
        if i == length - 1 or tran.reset:
            for j in range(min_value, max_value + 1):
                weight[i][j] = 1
        else:
            for j in range(min_value, max_value + 1):
                weight[i][j] = 0
                for k, w in weight[i + 1].items():
                    if k >= j:
                        weight[i][j] += w

    # Now sample according to the weights
    double_times = []
    cur_time = 0
    for i in range(length):
        start_time = max(min_time[i], cur_time)
        distr = []
        for j in range(start_time, max_time[i] + 1):
            distr.append(weight[i][j])
        if sum(distr) == 0:
            return None  # sampling failed
        cur_time = sample_distribution(distr) + start_time
        double_times.append(cur_time)
        if path[i].reset:
            cur_time = 0

    # Finally, change doubled time to fractions.
    ltw = []
    for i in range(length):
        if double_times[i] % 2 == 0:
            time = double_times[i] // 2
        else:
            time = double_times[i] // 2 + 0.5
        ltw.append(TimedWord(path[i].action, time))

    # Convert logical-timed word to delayed-timed word.
    dtw = []
    for i in range(length):
        if i == 0 or path[i - 1].reset:
            dtw.append(TimedWord(path[i].action, ltw[i].time))
        else:
            dtw.append(TimedWord(path[i].action, ltw[i].time - ltw[i - 1].time))
    return dtw


# 测试执行
def test_execution(hypothesis, system, sample):
    system_res, real_value = system.test_DTWs(sample)
    hypothesis_res, value = hypothesis.test_DTWs(sample)
    return real_value != value


# --------------------------------- auxiliary function ---------------------------------

def get_random_delay(upper_guard):
    time = random.randint(0, upper_guard * 3 + 1)
    if time % 2 == 0:
        time = time // 2
    else:
        time = time // 2 + 0.5
    return time


#  Get the double of the minimal number in an interval.
def min_constraint_double(c):
    """
        1. if the interval is empty, return None
        2. if [a, b$, return "2 * a".
        3. if (a, b$, return "2 * a + 1".
    """
    if c.is_empty():
        return None
    if c.closed_min:
        return int(2 * float(c.min_value))
    else:
        return int(2 * float(c.min_value) + 1)


#  Get the double of the maximal number in an interval.
def max_constraint_double(c, upperGuard):
    """
        1. if the interval is empty, return None
        2. if $a, b], return "2 * b".
        3. if $a, b), return "2 * b - 1".
        4. if $a, +), return "2 * upperGuard + 1".
    """
    if c.is_empty():
        return None
    if c.closed_max:
        return int(2 * float(c.max_value))
    elif c.max_value == '+':
        return int(2 * upperGuard + 1)
    else:
        return int(2 * float(c.max_value) - 1)


def sample_distribution(distr):
    s = sum(distr)
    if s == 0:
        return None
    a = random.randint(0, s - 1)
    for i, n in enumerate(distr):
        if n > a:
            return i
        else:
            a -= n

import random
import queue
import copy
from common.TimedWord import TimedWord
from smart_learning.mutation import mutant_generation, mutant_sample, NFA_mutant, mutation_analysis, test_selection


def mutation_testing(hypothesisOTA, upper_guard, state_num, system):
    # 生成候选测试集
    test_num = 2000  # 候选测试集数量
    tests = []
    pretry = 0.9
    pstop = 0.02
    linfix = int(state_num / 2)
    max_steps = 2 * state_num
    for i in range(test_num):
        tests.append(test_case_generation(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    # 生成变异体及变异分析
    Tsel = []
    cMuts = []
    nacc = 4
    nsel = 100
    mutations = mutant_generation(hypothesisOTA, upper_guard, nacc)
    if len(mutations) > 0:
        IMutsel = mutant_sample(hypothesisOTA.states, mutations)
        NMut = NFA_mutant(hypothesisOTA, IMutsel)
        pre_tests = []
        for t0 in tests:
            cMut = mutation_analysis(t0, NMut)
            if cMut:
                pre_tests.append(t0)
                cMuts.append(cMut)
        if cMuts:
            Tsel = test_selection(pre_tests, IMutsel, cMuts, nsel)  # 选择尽可能少的测试集覆盖cMuts里所包含的muts

    # 测试执行
    if len(Tsel) > 0:
        equivalent, ctx = test_execution(hypothesisOTA, system, Tsel)
    else:
        raise Exception("Mutation Failed!")
    return equivalent, ctx


# 测试序列生成(根据hypothesis)
def test_case_generation(hypothesis, pretry, pstop, max_steps, linfix, upper_guard):
    tests = []
    state = hypothesis.init_state
    li = random.randint(1, linfix)
    now_time = 0
    if coin_flip(0.5):
        actions = []
        for i in range(li):
            actions.append(random.sample(hypothesis.actions, 1))
        for action in actions:
            time = random.randint(0, upper_guard * 2 + 1)
            if time % 2 == 0:
                time = time // 2
            else:
                time = time // 2 + 0.5
            temp_DTW = TimedWord(action, time)
            temp_LTW = TimedWord(action, now_time + time)
            tests.append(temp_DTW)
            for t in hypothesis.trans:
                if t.source == state and t.is_passing_tran(temp_LTW):
                    state = t.target
                    if t.reset:
                        now_time = 0
                    else:
                        now_time = temp_LTW.time
                    break

    state_choice = hypothesis.states
    state_choice.remove(state)

    while True:
        if state_choice:
            rS = random.choice(state_choice)
        else:
            rS = state
        rI = random.choice(hypothesis.actions)
        time = random.randint(0, upper_guard * 2 + 1)
        if time % 2 == 0:
            time = time // 2
        else:
            time = time // 2 + 0.5
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
                rSteps_i.append(random.sample(hypothesis.actions, 1))
            rSteps = []
            for rsi in rSteps_i:
                time = random.randint(0, upper_guard * 2 + 1)
                if time % 2 == 0:
                    time = time // 2
                else:
                    time = time // 2 + 0.5
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
            tests = tests + p0 + [rI_temp_DTW] + rSteps
            if len(tests) > max_steps:
                break
            elif coin_flip(pstop):
                break
        elif coin_flip(1 - pretry):
            break
    return tests


# 测试执行
def test_execution(hypothesisOTA, system, tests):
    flag = True
    ctx = []
    for test in tests:
        DRTWs, value = hypothesisOTA.test_DTWs(test)
        realDRTWs, realValue = system.test_DTWs(test)
        if realValue != value:
            flag = False
            ctx = test
            break
    return flag, ctx


# --------------------------------- auxiliary function ---------------------------------

def coin_flip(p):
    return random.random() <= p


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
                time = random.randint(0, upper_guard * 2 + 1)
                if time % 2 == 0:
                    time = time // 2
                else:
                    time = time // 2 + 0.5
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

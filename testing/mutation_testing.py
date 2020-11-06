import random
import copy
import math
from itertools import product
from common.TimedWord import TimedWord
from common.hypothesis import OTATran
from common.TimeInterval import Guard, guard_split
from testing.random_testing import test_generation_2


class Mut(object):
    def __init__(self, mId, tran_traces):
        self.mId = mId
        self.tran_traces = tran_traces


class NOTA(object):
    def __init__(self, actions, states, init_state, trans, f_states):
        self.actions = actions
        self.states = states
        self.init_state = init_state
        self.trans = trans
        self.f_states = f_states


class NOTATran(object):
    def __init__(self, tran_id, source, action, guards, reset, target, is_mut):
        self.tran_id = tran_id
        self.source = source
        self.action = action
        self.guards = guards
        self.reset = reset
        self.target = target
        self.is_mut = is_mut

    def is_passing_tran(self, ltw):
        if ltw.action == self.action:
            for guard in self.guards:
                if guard.is_in_interval(ltw.time):
                    return True
        else:
            return False
        return False


# 变异测试主函数
def mutation_testing(hypothesisOTA, upper_guard, state_num, system):
    equivalent = True
    ctx = None

    # 参数配置 - 测试生成
    pretry = 0.9
    pstop = 0.02
    max_steps = min(int(2 * state_num), int(2 * len(hypothesisOTA.states)))
    linfix = min(math.ceil(len(hypothesisOTA.states) / 2), math.ceil(state_num / 2))
    test_num = int(len(hypothesisOTA.states) * len(hypothesisOTA.actions) * upper_guard * 10)
    # 参数配置 - 变异相关
    region_num = system.get_minimal_duration()  # It can also be set by the user.
    nacc = 10
    k = 1
    nsel = 200

    # 测试集生成
    tests = []
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    # step1: time变异分析
    guard_tests = mutation_timed(hypothesisOTA, region_num, upper_guard, tests)
    # 测试执行
    if len(guard_tests) > 0:
        print('guard mutation', len(guard_tests))
        equivalent, ctx = test_execution(hypothesisOTA, system, guard_tests)

    # step2: 如果未找到反例
    if equivalent:
        # state 变异分析
        state_tests = mutation_state(hypothesisOTA, guard_tests, state_num, nacc, k, region_num, upper_guard, tests)
        # 测试执行
        if len(state_tests) > 0:
            print('state mutation', len(state_tests))
            equivalent, ctx = test_execution(hypothesisOTA, system, state_tests)

        # # step3: 随机选取测试集直到数量满足nsel
        # if equivalent and len(guard_tests) + len(state_tests) < nsel:
        #     rest_num = nsel - len(guard_tests) + len(state_tests)
        #     random_tests = random_select(tests, rest_num, guard_tests + state_tests)
        #     print('random tests', len(random_tests))
        #     equivalent, ctx = test_execution(hypothesisOTA, system, random_tests)

    return equivalent, ctx


# timed 变异
def mutation_timed(hypothesis, region_num, upper_guard, tests):
    Tsel = []
    # 生成变异体
    mutations = timed_mutation_generation(hypothesis, region_num, upper_guard)
    print('timed_mutations', len(mutations))
    # 生成NOTA
    timed_NOTA = NOTA_generation(hypothesis, mutations, [], region_num, upper_guard)
    print('timed_NOTA trans', len(timed_NOTA.trans))
    # 变异分析
    print('now timed_mutation_analysis')
    tran_dict = get_tran_dict(timed_NOTA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = timed_mutation_analysis(timed_NOTA, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    # 测试筛选
    if C_tests:
        Tsel = test_selection([], tests_valid, C, C_tests)
    return Tsel


# 生成变异体 - timed变异算子
def timed_mutation_generation(hypothesis, region_num, upper_guard):
    new_trans = []
    tran_id = len(hypothesis.trans)
    for tran in hypothesis.trans:
        for guard in tran.guards:
            guard_min = guard.get_min()
            guard_max = guard.get_max()
            # 特殊情况处理 - [0,+)
            if guard_min == 0 and guard.get_closed_min() and guard_max == float("inf"):
                temp_guards = guard_split(guard, region_num, upper_guard)
                for state in hypothesis.states:
                    for temp_guard in temp_guards:
                        if state == tran.target:
                            new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], not tran.reset, state))
                            tran_id += 1
                        else:
                            new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], tran.reset, state))
                            tran_id += 1
                            new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], not tran.reset, state))
                            tran_id += 1
                continue
                # 正常情况处理 - 处理左边
            if guard_min == 0:
                if not guard.get_closed_min():
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [Guard("[0,0]")], tran.reset, tran.target))
                    tran_id += 1
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [Guard("[0,0]")], not tran.reset, tran.target))
                    tran_id += 1
            else:
                if guard.get_closed_min():
                    left_guard = Guard('[0,' + str(guard_min) + ')')
                else:
                    left_guard = Guard('[0,' + str(guard_min) + ']')
                temp_guards = guard_split(left_guard, region_num, upper_guard)
                for temp_guard in temp_guards:
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    tran_id += 1
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
                    tran_id += 1
            # 正常情况处理 - 处理右边
            if guard_max == float("inf") or guard_max >= upper_guard:
                pass
            else:
                if guard.get_closed_max():
                    left_guard = Guard('(' + str(guard_max) + ',+)')
                else:
                    left_guard = Guard('[' + str(guard_max) + ',+)')
                temp_guards = guard_split(left_guard, region_num, upper_guard)
                for temp_guard in temp_guards:
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    tran_id += 1
                    new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
                    tran_id += 1
            # 正常情况处理 - 处理自身
            temp_guards = guard_split(guard, region_num, upper_guard)
            for temp_guard in temp_guards:
                new_trans.append(OTATran(tran_id, tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
                tran_id += 1
    return new_trans


# timed变异分析 - 获得test覆盖到的mut
def timed_mutation_analysis(timed_NOTA, test, C, tran_dict):
    C_test = []
    cache = {}
    for i in range(len(test)):
        cache[i] = []

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return True
        cur_time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, cur_time)
        cur_trans = tran_dict[state]
        for tran in cur_trans:
            if tran.is_passing_tran(new_LTW):
                if tran.reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if tran.is_mut:
                    if tran.tran_id not in C_test:
                        C_test.append(tran.tran_id)
                    if tran.tran_id not in C:
                        C.append(tran.tran_id)
                if [tran.target, tempTime] not in cache[test_index]:
                    cache[test_index].append([tran.target, tempTime])
                    tree_create(tran.target, tempTime, test_index + 1)

    tree_create(timed_NOTA.init_state, 0, 0)
    return C_test, C


# state 变异
def mutation_state(hypothesis, guard_tests, state_num, nacc, k, region_num, upper_guard, tests):
    Tsel = []
    # 生成变异体
    mutations = split_state_mutation_generation(hypothesis, nacc, k, state_num)
    print('state_mutations', len(mutations))
    # 生成NFA
    # # mut由于数量较大，因此可进行一定的筛选
    # mutations = mutant_sample()
    state_NOTA = NOTA_generation(hypothesis, [], mutations, region_num, upper_guard)
    print('state_NOTA trans', len(state_NOTA.trans))
    # 变异分析
    print('now state_mutation_analysis')
    tran_dict = get_tran_dict(state_NOTA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = split_state_mutation_analysis(state_NOTA, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    # 测试筛选
    if C_tests:
        Tsel = test_selection(guard_tests, tests_valid, C, C_tests)
    return Tsel


# 生成变异体 - split-state算子
def split_state_mutation_generation(hypothesis, nacc, k, state_num):
    Muts = []
    mId = 0
    for state in hypothesis.states:
        set_accq = get_all_acc(hypothesis, state, state_num)
        if len(set_accq) < 2:
            continue
        elif nacc >= len(set_accq):
            subset_accq = set_accq
        else:
            subset_accq = random.sample(set_accq, nacc)
        for s1 in subset_accq:
            for s2 in subset_accq:
                if s1 == s2:
                    continue
                else:
                    mut, mId = split_state_operator(s1, s2, k, hypothesis, mId)
                    if mut is not None:
                        Muts.extend(mut)
    return Muts


# split-state算子
def split_state_operator(s1, s2, k, hypothesis, mId):
    if len(s1) < len(s2) and s2[0:len(s1)] == s1:
        return None, mId
    suffix = arg_maxs(s1, s2)
    prefix = s1[0:len(s1) - len(suffix)]
    if len(prefix) == 0:
        return None, mId
    p = prefix[len(prefix) - 1]
    mutants = []
    trans_list = k_step_trans(hypothesis, s1[-1].target, k)
    for distSeq in trans_list:
        mut_tran = [p] + suffix + distSeq
        mutants.append(Mut(mId, mut_tran))
        mId += 1
    return mutants, mId


# state变异分析 - 获得test覆盖到的mut
def split_state_mutation_analysis(state_NOTA, test, C, tran_dict):
    C_test = []

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return True
        cur_time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, cur_time)
        if state not in tran_dict.keys():
            return True
        cur_trans = tran_dict[state]
        for tran in cur_trans:
            if tran.is_passing_tran(new_LTW):
                if tran.reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if tran.target in state_NOTA.f_states:
                    mId = tran.target.split('_')[0]
                    if mId not in C_test:
                        C_test.append(mId)
                    if mId not in C:
                        C.append(mId)
                tree_create(tran.target, tempTime, test_index + 1)

    tree_create(state_NOTA.init_state, 0, 0)
    return C_test, C


# 生成NOTA
def NOTA_generation(hypothesis, timed_muts, split_state_muts, region_num, upper_guard):
    hypothesis = copy.deepcopy(hypothesis)
    states = hypothesis.states
    init_state = hypothesis.init_state
    actions = hypothesis.actions
    trans = []
    f_states = []
    for tran in hypothesis.trans:
        trans.append(NOTATran(tran.tran_id, tran.source, tran.action, tran.guards, tran.reset, tran.target, False))

    # 处理timed变异体
    for timed_mut in timed_muts:
        trans.append(NOTATran(timed_mut.tran_id, timed_mut.source, timed_mut.action, timed_mut.guards, timed_mut.reset, timed_mut.target, True))

    # 处理状态分割变异体
    x = 0
    for split_state_mut in split_state_muts:
        cache_trans = []
        for tran in split_state_mut.tran_traces:
            new_trans = split_tran_guard(tran, region_num, upper_guard)
            cache_trans.append(new_trans)
        trans_list = [new_trans for new_trans in product(*cache_trans)]
        mId = 0
        for one_trans in trans_list:
            source_state = one_trans[0].source
            target_state = None
            for tran in one_trans:
                tran = copy.deepcopy(tran)
                target_state = str(mId) + '_' + str(x)
                tran.source = source_state
                tran.target = target_state
                trans.append(tran)
                states.append(target_state)
                x += 1
                source_state = target_state
            mId += 1
            f_states.append(target_state)
    return NOTA(actions, states, init_state, trans, f_states)


# 测试筛选
def test_selection(cache_tests, Tests, C, C_tests):
    Tsel = []
    CC = copy.deepcopy(C)  # all mutations
    tests = copy.deepcopy(Tests)  # tests
    cset = copy.deepcopy(C_tests)  # tests 对应的 cover mutation set
    pre_set = []
    while CC:
        cur_index = 0
        cur_max = []
        for i in range(len(cset)):
            cset[i] = list(set(cset[i]).difference(set(pre_set)))
            if len(cur_max) < len(cset[i]):
                cur_max = cset[i]
                cur_index = i
        if cur_max:
            if tests[cur_index] not in cache_tests:
                Tsel.append(tests[cur_index])
            pre_set = cur_max
        else:
            break
    return Tsel


# 测试执行
def test_execution(hypothesisOTA, system, tests):
    flag = True
    ctx = []
    for test in tests:
        test_list = prefixes(test)
        for j in test_list:
            DRTWs, value = hypothesisOTA.test_DTWs(j)
            realDRTWs, realValue = system.test_DTWs(j)
            if realValue != value:
                flag = False
                ctx = test
                return flag, ctx
    return flag, ctx


# --------------------------------- auxiliary function --------------------------------

# 在tests中随机选择test_num个与cur_tests不重复元素
def random_select(tests, test_num, cur_tests):
    for test in cur_tests:
        tests.remove(test)
    if len(tests) < test_num:
        return tests
    else:
        return random.sample(tests, test_num)


# 将NOTA中的迁移按source分组
def get_tran_dict(NOTA_mut):
    tran_dict = {}
    for tran in NOTA_mut.trans:
        if tran.source in tran_dict.keys():
            tran_dict[tran.source].append(tran)
        else:
            tran_dict[tran.source] = [tran]
    return tran_dict


# 前缀集
def prefixes(tws):
    new_prefixes = []
    for i in range(1, len(tws) + 1):
        temp_tws = tws[:i]
        new_prefixes.append(temp_tws)
    return new_prefixes


# 找到s1和s2的最长公共后缀
def arg_maxs(s1, s2):
    ts = []
    if len(s1) < len(s2):
        min_test = s1
    else:
        min_test = s2
    for i in range(len(min_test)):
        if not s1[-1 - i].tran_id == s2[-1 - i].tran_id:
            break
        ts = min_test[(len(min_test) - 1 - i):]
    return ts


# 找到qs状态后走step的所有路径
def k_step_trans(hypothesis, qs, k):
    trans_list = []

    def recursion(cur_state, paths):
        if len(paths) == k:
            if paths not in trans_list:
                trans_list.append(paths)
            return True
        for tran in hypothesis.trans:
            if tran.source == cur_state:
                if tran.target == cur_state and len(paths) >= 1:
                    if paths[-1] == tran:
                        continue
                recursion(tran.target, copy.deepcopy(paths) + [tran])

    recursion(qs, [])
    return trans_list


# 将迁移的guard分割
def split_tran_guard(tran, region_num, upper_guard):
    trans = []
    for guard in tran.guards:
        temp_guards = guard_split(guard, region_num, upper_guard)
        for temp_guard in temp_guards:
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
    return trans


# get mutated access seq leading to a single state
def get_all_acc(hypothesis, state, state_num):
    paths = []
    max_path_length = min(int(len(hypothesis.states) * 1.5), state_num * 1.5)

    def get_next_tran(sn, path):
        if len(path) > max_path_length:
            return True
        if sn == state and path:
            if path not in paths:
                paths.append(path)
            return True
        for tran in hypothesis.trans:
            if tran.source == sn:
                if tran.target == sn and len(path) >= 2:
                    if path[-1].source == sn and path[-2].source == sn:
                        continue
                get_next_tran(tran.target, copy.deepcopy(path) + [tran])
        return True

    get_next_tran(hypothesis.init_state, [])
    return paths

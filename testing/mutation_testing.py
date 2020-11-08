import math
import copy
import random
from itertools import product
from testing.random_testing import test_generation_2
from common.hypothesis import OTATran
from common.TimedWord import TimedWord
from common.TimeInterval import Guard, guard_split


class NFA(object):
    def __init__(self, states, init_state, actions, trans, final_states):
        self.states = states
        self.init_state = init_state
        self.actions = actions
        self.trans = trans
        self.final_states = final_states


# 基于变异的测试主函数
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
    nacc = 8
    k = 1
    nsel = 200

    # 测试集生成
    tests = []
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    tested = []  # 缓存已测试序列
    # step1: timed变异
    timed_tests = mutation_timed(hypothesisOTA, region_num, upper_guard, k, tests)
    if len(timed_tests) > 0:
        print('number of timed tests', len(timed_tests))
        equivalent, ctx = test_execution(hypothesisOTA, system, timed_tests)
        tested = timed_tests

    # step2: 如果未找到反例, transition变异
    if equivalent:
        tran_tests = mutation_tran(hypothesisOTA, k, region_num, upper_guard, tests)
        if len(tran_tests) > 0:
            tran_tests = remove_tested(tran_tests, tested)
            print('number of tran tests', len(tran_tests))
            equivalent, ctx = test_execution(hypothesisOTA, system, tran_tests)
            tested += tran_tests

        # step3: 如果未找到反例, state变异
        if equivalent:
            state_tests = mutation_state(hypothesisOTA, state_num, nacc, k, region_num, upper_guard, tests)
            if len(state_tests) > 0:
                state_tests = remove_tested(state_tests, tested)
                print('number of state tests', len(state_tests))
                equivalent, ctx = test_execution(hypothesisOTA, system, state_tests)
                tested += state_tests

            # step4: 随机选取测试集直到数量满足nsel
            if equivalent and len(timed_tests) + len(state_tests) + len(tran_tests) < nsel:
                tests = remove_tested(tests, tested)
                if nsel - len(timed_tests) - len(state_tests) - len(tran_tests) > len(tests):
                    random_tests = tests
                else:
                    random_tests = random.sample(tests, nsel - len(timed_tests) - len(state_tests) - len(tran_tests))
                print('number of random tests', len(random_tests))
                equivalent, ctx = test_execution(hypothesisOTA, system, random_tests)

    return equivalent, ctx


# timed mutation
def mutation_timed(hypothesis, region_num, upper_guard, k, tests):
    Tsel = []
    # 生成变异体
    mutations = timed_mutation_generation(hypothesis, region_num, upper_guard, k)
    print('number of timed_mutations', len(mutations))
    # 生成NFA
    muts_NFA = NFA_generation(mutations, hypothesis)
    print('number of timed NFA trans', len(muts_NFA.trans))
    # 变异分析
    print('Starting mutation analysis...')
    tran_dict = get_tran_dict(muts_NFA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = mutation_analysis(muts_NFA, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    # 测试筛选
    if C_tests:
        Tsel = test_selection(tests_valid, C, C_tests)
    return Tsel


# timed mutation generation/operator
def timed_mutation_generation(hypothesis, region_num, upper_guard, k):
    mutations = []
    new_trans = []
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
                            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, state))
                        else:
                            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, state))
                            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, state))
                continue
            # 正常情况处理 - 处理左边
            if guard_min == 0:
                if not guard.get_closed_min():
                    new_trans.append(OTATran('', tran.source, tran.action, [Guard("[0,0]")], tran.reset, tran.target))
                    new_trans.append(OTATran('', tran.source, tran.action, [Guard("[0,0]")], not tran.reset, tran.target))
            else:
                if guard.get_closed_min():
                    left_guard = Guard('[0,' + str(guard_min) + ')')
                else:
                    left_guard = Guard('[0,' + str(guard_min) + ']')
                temp_guards = guard_split(left_guard, region_num, upper_guard)
                for temp_guard in temp_guards:
                    new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
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
                    new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
            # 正常情况处理 - 处理自身
            temp_guards = guard_split(guard, region_num, upper_guard)
            for temp_guard in temp_guards:
                new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
    # 每个迁移都往后走k步
    for new_tran in new_trans:
        suffixes = k_step_trans(hypothesis, new_tran.target, k)
        for suffix in suffixes:
            mutations.append([new_tran] + suffix)
    return mutations


# split_state mutation
def mutation_state(hypothesis, state_num, nacc, k, region_num, upper_guard, tests):
    Tsel = []
    # 生成变异体
    mutations = split_state_mutation_generation(hypothesis, nacc, k, state_num, region_num, upper_guard)
    print('number of split_state_mutations', len(mutations))
    # 生成NFA
    muts_NFA = NFA_generation(mutations, hypothesis)
    print('number of state NFA trans', len(muts_NFA.trans))
    # 变异分析
    print('Starting mutation analysis...')
    tran_dict = get_tran_dict(muts_NFA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = mutation_analysis(muts_NFA, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    # 测试筛选
    if C_tests:
        Tsel = test_selection(tests_valid, C, C_tests)
    return Tsel


# split-state mutation generation
def split_state_mutation_generation(hypothesis, nacc, k, state_num, region_num, upper_guard):
    mutations = []
    temp_mutations = []
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
                    muts = split_state_operator(s1, s2, k, hypothesis)
                    if muts is not None:
                        temp_mutations.extend(muts)
    for temp_mut in temp_mutations:
        cache_trans = []
        for i in range(len(temp_mut)):
            if i == 0:
                new_trans = split_tran_guard(temp_mut[i], region_num, upper_guard)
            elif i >= len(temp_mut) - k:
                new_trans = split_tran_guard(temp_mut[i], region_num, upper_guard)
            else:
                new_trans = [temp_mut[i]]
            cache_trans.append(new_trans)
        trans_list = [new_trans for new_trans in product(*cache_trans)]
        mutations.extend(trans_list)
    return mutations


# split-state operator
def split_state_operator(s1, s2, k, hypothesis):
    if len(s1) < len(s2) and s2[0:len(s1)] == s1:
        return None
    suffix = arg_maxs(s1, s2)
    prefix = s1[0:len(s1) - len(suffix)]
    if len(prefix) == 0:
        return None
    p_tran = prefix[len(prefix) - 1]
    mutants = []
    trans_list = k_step_trans(hypothesis, s1[-1].target, k)
    for distSeq in trans_list:
        mut_tran = [p_tran] + suffix + distSeq
        mutants.append(mut_tran)
    return mutants


# transition mutation
def mutation_tran(hypothesis, k, region_num, upper_guard, tests):
    Tsel = []
    # 生成变异体
    mutations = tran_mutation_generation(hypothesis, k, region_num, upper_guard)
    print('number of tran_mutations', len(mutations))
    # 生成NFA
    muts_NFA = NFA_generation(mutations, hypothesis)
    print('number of tran NFA trans', len(muts_NFA.trans))
    # 变异分析
    print('Starting mutation analysis...')
    tran_dict = get_tran_dict(muts_NFA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = mutation_analysis(muts_NFA, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    # 测试筛选
    if C_tests:
        Tsel = test_selection(tests_valid, C, C_tests)
    return Tsel


# tran mutation generation
def tran_mutation_generation(hypothesis, k, region_num, upper_guard):
    mutations = []
    step_trans_dict = {}
    for state in hypothesis.states:
        step_trans_dict[state] = k_step_trans(hypothesis, state, k)
    for tran in hypothesis.trans:
        trans = split_tran_guard_remove_first(tran, region_num, upper_guard)
        for state in hypothesis.states:
            if state != tran.target:
                for prefix in trans:
                    temp = copy.deepcopy(prefix)
                    temp.target = state
                    for suffix in step_trans_dict[state]:
                        mutations.append([temp] + suffix)
    return mutations


# generation NFA-based mutant representation
def NFA_generation(mutations, hypothesis):
    hypothesis = copy.deepcopy(hypothesis)
    states = hypothesis.states
    init_state = hypothesis.init_state
    actions = hypothesis.actions
    trans = hypothesis.trans
    final_states = []
    mId = 0
    for mutation in mutations:
        count = 0
        source_state = mutation[0].source
        target_state = None
        for tran in mutation:
            tran = copy.deepcopy(tran)
            target_state = str(mId) + '_' + str(count)
            tran.source = source_state
            tran.target = target_state
            trans.append(tran)
            states.append(target_state)
            count += 1
            source_state = target_state
        mId += 1
        final_states.append(target_state)
    return NFA(states, init_state, actions, trans, final_states)


# mutation analysis for single test using NFA-based mutant representation
def mutation_analysis(muts_NFA, test, C, tran_dict):
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
                if tran.target in muts_NFA.final_states:
                    mId = tran.target.split('_')[0]
                    if mId not in C_test:
                        C_test.append(mId)
                    if mId not in C:
                        C.append(mId)
                tree_create(tran.target, tempTime, test_index + 1)

    tree_create(muts_NFA.init_state, 0, 0)
    return C_test, C


# 测试筛选
def test_selection(Tests, C, C_tests):
    Tsel = []
    c = copy.deepcopy(C)  # all mutations
    tests = copy.deepcopy(Tests)  # tests
    cset = copy.deepcopy(C_tests)  # tests 对应的 cover mutation set
    pre_set = []
    while c:
        cur_index = 0
        cur_max = []
        for i in range(len(cset)):
            cset[i] = list(set(cset[i]).difference(set(pre_set)))
            if len(cur_max) < len(cset[i]):
                cur_max = cset[i]
                cur_index = i
        if cur_max:
            Tsel.append(tests[cur_index])
            pre_set = cur_max
        else:
            break
    return Tsel


# 测试执行
def test_execution(hypothesis, system, tests):
    flag = True
    ctx = []
    for test in tests:
        test_list = prefixes(test)
        for j in test_list:
            DRTWs, value = hypothesis.test_DTWs(j)
            realDRTWs, realValue = system.test_DTWs(j)
            if realValue != value:
                flag = False
                ctx = test
                return flag, ctx
    return flag, ctx


# --------------------------------- auxiliary function --------------------------------

# tests中删除cur_tests
def remove_tested(tests, cur_tests):
    for test in cur_tests:
        if test in tests:
            tests.remove(test)
    return tests


# 前缀集
def prefixes(tws):
    new_prefixes = []
    for i in range(1, len(tws) + 1):
        temp_tws = tws[:i]
        new_prefixes.append(temp_tws)
    return new_prefixes


# 找到qs状态后走step的所有路径
def k_step_trans(hypothesis, q, k):
    trans_list = []

    def recursion(cur_state, paths):
        if len(paths) == k:
            if paths not in trans_list:
                trans_list.append(paths)
            return True
        for tran in hypothesis.trans:
            if tran.source == cur_state:
                if len(paths) > 0 and paths[-1] == tran:
                    continue
                recursion(tran.target, copy.deepcopy(paths) + [tran])

    recursion(q, [])
    return trans_list


# 将NFA中的迁移按source分组
def get_tran_dict(muts_NFA):
    tran_dict = {}
    for tran in muts_NFA.trans:
        if tran.source in tran_dict.keys():
            tran_dict[tran.source].append(tran)
        else:
            tran_dict[tran.source] = [tran]
    return tran_dict


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
        for tran in hypothesis.trans:
            if tran.source == sn:
                if len(path) > 0 and tran == path[-1]:
                    continue
                if len(path) > 1 and tran.target == sn:
                    if path[-1].source == sn and path[-2].source == sn:
                        continue
                get_next_tran(tran.target, copy.deepcopy(path) + [tran])

    get_next_tran(hypothesis.init_state, [])
    return paths


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


# 将迁移的guard分割
def split_tran_guard(tran, region_num, upper_guard):
    trans = []
    for guard in tran.guards:
        temp_guards = guard_split(guard, region_num, upper_guard)
        for temp_guard in temp_guards:
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
    return trans


# 将迁移的guard分割 - 不取切分的第一个
def split_tran_guard_remove_first(tran, region_num, upper_guard):
    trans = []
    for guard in tran.guards:
        temp_guards = guard_split(guard, region_num, upper_guard)[1:]
        for temp_guard in temp_guards:
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
    return trans

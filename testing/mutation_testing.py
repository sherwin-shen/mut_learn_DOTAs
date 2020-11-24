import random
from copy import deepcopy
from itertools import product
from common.TimedWord import TimedWord
from common.hypothesis import OTATran
from common.TimeInterval import Guard, guard_split
from testing.random_testing import test_generation_4, test_generation_5


class NFA(object):
    def __init__(self, states, init_state, actions, trans, sink_state, final_states):
        self.states = states
        self.init_state = init_state
        self.actions = actions
        self.trans = trans
        self.sink_state = sink_state
        self.final_states = final_states


# 基于变异的测试主函数
def mutation_testing(hypothesisOTA, upper_guard, state_num, system, prectxs):
    equivalent = True
    ctx = None

    # 参数配置 - 测试生成
    pretry = 0.9
    pstop = 0.05
    pvalid = 0.8
    pnext = 0.8
    max_steps = min(int(2 * state_num), int(2 * len(hypothesisOTA.states)))
    test_num = int(len(hypothesisOTA.states) * len(hypothesisOTA.actions) * upper_guard * 10)

    # 参数配置 - 变异相关
    region_num = int((system.get_minimal_duration()+1)/2)  # It can also be set by the user.
    region_num = int(upper_guard/5)
    nacc = 8
    k = 1
    nsel = 200

    for c in prectxs:
        if len(c) >= max_steps:
            prectxs.remove(c)
    # 测试集生成
    tests = []
    for i in range(test_num):
        if prectxs:
            prectx = random.choice(prectxs)
        else:
            prectx = []
        tests.append(test_generation_5(hypothesisOTA, pretry, pstop, max_steps, pvalid, pnext, upper_guard, prectx))

    tested = []  # 缓存已测试序列

    '''
    mutation_tests = mutation_timed(hypothesisOTA, region_num, upper_guard, k, tests)
    mutation_tests = mutation_tran(hypothesisOTA, k, region_num, upper_guard, tests)
    mutation_tests = mutation_state(hypothesisOTA, state_num, nacc, k, region_num, upper_guard, tests)
    if len(mutation_tests) > 0:
        print('number of timed tests', len(mutation_tests))
        equivalent, ctx = test_execution(hypothesisOTA, system, mutation_tests)
        tested = mutation_tests
        # step4: 随机选取测试集直到数量满足nsel
    if equivalent and len(mutation_tests) < nsel:
        tests = remove_tested(tests, tested)
        if nsel - len(mutation_tests) > len(tests):
            random_tests = tests
        else:
            random_tests = random.sample(tests, nsel - len(mutation_tests))
        print('number of random tests', len(random_tests))
        equivalent, ctx = test_execution(hypothesisOTA, system, random_tests)
    '''

    #######################################################

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

            '''
            # step4: 随机选取测试集直到数量满足nsel
                if equivalent and len(timed_tests) + len(state_tests) + len(tran_tests) < nsel:
                    tests = remove_tested(tests, tested)
                    if nsel - len(timed_tests) - len(state_tests) - len(tran_tests) > len(tests):
                        random_tests = tests
                    else:
                        random_tests = random.sample(tests, nsel - len(timed_tests) - len(state_tests) - len(tran_tests))
                    print('number of random tests', len(random_tests))
                    equivalent, ctx = test_execution(hypothesisOTA, system, random_tests)
            '''

    ###########################################################################################################

    return equivalent, ctx


# timed mutation
def mutation_timed(hypothesis, region_num, upper_guard, k, tests):
    Tsel = []
    # 生成变异体
    mutations, new_trans = timed_mutation_generation(hypothesis, region_num, upper_guard, k)
    print('number of timed_mutations', len(mutations))
    # 生成NFA
    muts_NFA = NFA_generation_tran(new_trans, hypothesis)
    #muts_NFA = NFA(hypothesis.states, hypothesis.init_state, hypothesis.actions, hypothesis.trans + new_trans, hypothesis.sink_state, hypothesis.accept_states)
    print('number of timed NFA trans', len(muts_NFA.trans))
    # 变异分析
    print('Starting mutation analysis...')
    tran_dict = get_tran_dict(muts_NFA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = mutation_analysis_tran(muts_NFA, hypothesis, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    if C:
        cover_rate = float(len(C)) / float(len(new_trans))
        print("tests cover rate:", cover_rate)
    # 测试筛选
    if len(C_tests) > 10:
        Tsel = test_selection(tests_valid, C, C_tests)
    else:
        Tsel = tests_valid
    return Tsel


# timed mutation generation/operator
def timed_mutation_generation(hypothesis, region_num, upper_guard, k):
    mutations = []
    new_trans = []
    mut_num = 0
    for tran in hypothesis.trans:
        if tran.source == hypothesis.sink_state and tran.target == hypothesis.sink_state:
            continue
        for guard in tran.guards:
            guard_min = guard.get_min()
            guard_max = guard.get_max()
            # 特殊情况处理 - [0,+)
            #if guard_min == 0 and guard.get_closed_min() and guard_max == float("inf"):
                #temp_guards = guard_split(guard, region_num, upper_guard)
                #for state in hypothesis.states:
                #    for temp_guard in temp_guards:
                #        if state == tran.target:
                #            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, state))
                #        else:
                #            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, state))
                #            new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, state))
                #continue
            # 正常情况处理 - 处理左边
            if guard_min == 0:
                if not guard.get_closed_min():
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [Guard("[0,0]")], tran.reset, tran.target))
                    mut_num += 1
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [Guard("[0,0]")], not tran.reset, tran.target))
                    mut_num += 1
            else:
                if guard.get_closed_min():
                    left_guard = Guard('[0,' + str(guard_min) + ')')
                else:
                    left_guard = Guard('[0,' + str(guard_min) + ']')
                temp_guards = guard_split(left_guard, region_num, upper_guard)
                if not temp_guards:
                    new_trans.append(OTATran('tran' + str(mut_num), tran.source, tran.action, [left_guard], not tran.reset, tran.target))
                    mut_num += 1
                for temp_guard in temp_guards:
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    mut_num += 1
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
                    mut_num += 1
            # 正常情况处理 - 处理右边
            if guard_max == float("inf") or guard_max >= upper_guard:
                pass
            else:
                if guard.get_closed_max():
                    left_guard = Guard('(' + str(guard_max) + ',+)')
                else:
                    left_guard = Guard('[' + str(guard_max) + ',+)')
                temp_guards = guard_split(left_guard, region_num, upper_guard)
                if not temp_guards:
                    new_trans.append(OTATran('tran' + str(mut_num), tran.source, tran.action, [left_guard], not tran.reset, tran.target))
                    mut_num += 1
                for temp_guard in temp_guards:
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    mut_num += 1
                    new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
                    mut_num += 1
            # 正常情况处理 - 处理自身
            new_trans.append(OTATran('tran'+str(mut_num), tran.source, tran.action, [guard], not tran.reset, tran.target))
            mut_num += 1
            #temp_guards = guard_split(guard, region_num, upper_guard)
            #for temp_guard in temp_guards:
            #    new_trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
    # 每个迁移都往后走k步
    #for new_tran in new_trans:
    #    suffixes = k_step_trans(hypothesis, new_tran.target, k)
    #    for suffix in suffixes:
    #        mutations.append([new_tran] + suffix)
    return mutations, new_trans


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
    #cover_rate = float(len(C)) / float(len(mutations))
    if C:
        cover_rate = float(len(C)) / float(len(mutations))
        print("tests cover rate:", cover_rate)
    # 测试筛选
    if C_tests:
        Tsel = test_selection(tests_valid, C, C_tests)
    return Tsel


# split-state mutation generation
def split_state_mutation_generation(hypothesis, nacc, k, state_num, region_num, upper_guard):
    mutations = []
    temp_mutations = []
    for state in hypothesis.states:
        if state == hypothesis.sink_state:
            continue
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
    '''
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
    '''
    #return mutations
    return temp_mutations


# split-state operator
def split_state_operator(s1, s2, k, hypothesis):
    #if len(s1) < len(s2) and s2[0:len(s1)] == s1:
    #    return None
    suffix = []
    #pre_action = hypothesis.init_state
    if not s1:
        p_tran = OTATran('', hypothesis.init_state, [], [], False, hypothesis.init_state)
    else:
        suffix = arg_maxs(s1, s2)
        prefix = s1[0:len(s1) - len(suffix)]
        if len(prefix) == 0:
            p_tran = OTATran('', hypothesis.init_state, [], [], False, hypothesis.init_state)
            #return None
        else:
            p_tran = prefix[len(prefix) - 1]
    mutants = []
    temp_tran = [p_tran] + suffix
    trans_list = k_step_trans(hypothesis, temp_tran[-1].target, k)
    for distSeq in trans_list:
        mut_tran = temp_tran + distSeq
        mutants.append(mut_tran)
    return mutants

# generation NFA-based mutant representation
def NFA_generation_tran(new_trans, hypothesis):
    hypothesis = deepcopy(hypothesis)
    states = hypothesis.states
    init_state = hypothesis.init_state
    actions = hypothesis.actions
    trans = hypothesis.trans
    sink_state = hypothesis.sink_state
    final_states = hypothesis.accept_states
    NFA_trans = []
    for tran in trans:
    #    if tran.target != sink_state:
        NFA_trans.append(tran)
    NFA_trans.extend(new_trans)
    return NFA(states, init_state, actions, NFA_trans, sink_state, final_states)
    #return NFA(hypothesis.states, hypothesis.init_state, hypothesis.actions, NFA_tran + new_trans, None, hypothesis.accept_states)


# generation NFA-based mutant representation
def NFA_generation(mutations, hypothesis):
    hypothesis = deepcopy(hypothesis)
    states = hypothesis.states
    init_state = hypothesis.init_state
    actions = hypothesis.actions
    trans = hypothesis.trans
    sink_state = hypothesis.sink_state
    final_states = []
    mId = 0
    for mutation in mutations:
        count = 0
        source_state = mutation[0].source
        target_state = None
        for tran in mutation:
            tran = deepcopy(tran)
            target_state = str(mId) + '_' + str(count)
            tran.source = source_state
            tran.target = target_state
            trans.append(tran)
            states.append(target_state)
            count += 1
            source_state = target_state
        mId += 1
        final_states.append(target_state)
    return NFA(states, init_state, actions, trans, sink_state, final_states)


# mutation analysis for single test using NFA-based mutant representation(transition mutation)
def mutation_analysis_tran(muts_NFA, hypothesis, test, C, tran_dict):
    C_test = []

    htran_dict = get_tran_dict(hypothesis)
    now_time = 0
    now_state = hypothesis.init_state
    test_result = []
    #if hypothesis.init_state in hypothesis.accept_states:
    #    test_result.append(1)
    #else:
    #    test_result.append(0)
    for t in test:
        temp_time = t.time + now_time
        new_LTW = TimedWord(t.action, temp_time)
        for tran in htran_dict[now_state]:
            if tran.is_passing_tran(new_LTW):
                now_state = tran.target
                if tran.reset:
                    now_time = 0
                else:
                    now_time = temp_time
                if now_state in hypothesis.accept_states:
                    test_result.append(1)
                elif now_state == hypothesis.sink_state:
                    test_result.append(-1)
                else:
                    test_result.append(0)

    def tree_create(state, preTime, test_index, mut_tran):
        if test_index >= len(test):
            return True
        cur_time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, cur_time)
        if state not in tran_dict.keys():
            return True
        if mut_tran:
            cur_trans = htran_dict[state]
            if state == mut_tran.source and mut_tran.is_passing_tran(new_LTW):
                if mut_tran.reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if mut_tran.target in muts_NFA.final_states:
                    state_flag = 1
                elif mut_tran.target == muts_NFA.sink_state:
                    state_flag = -1
                else:
                    state_flag = 0
                if state_flag != test_result[test_index]:
                    if mut_tran.tran_id not in C_test:
                        C_test.append(mut_tran.tran_id)
                    if mut_tran.tran_id not in C:
                        C.append(mut_tran.tran_id)
                    return True
                tree_create(mut_tran.target, tempTime, test_index + 1, mut_tran)
                return True
        else:
            cur_trans = tran_dict[state]
        #cur_trans = tran_dict[state]

        for tran in cur_trans:
            if tran.is_passing_tran(new_LTW):
                if tran.reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if tran.target in muts_NFA.final_states:
                    state_flag = 1
                elif tran.target == muts_NFA.sink_state:
                    state_flag = -1
                else:
                    state_flag = 0

                if isinstance(tran.tran_id, str):
                    mut_tran = tran

                if mut_tran:
                    if state_flag != test_result[test_index]:
                        if mut_tran.tran_id not in C_test:
                            C_test.append(mut_tran.tran_id)
                        if mut_tran.tran_id not in C:
                            C.append(mut_tran.tran_id)
                        return True
                    tree_create(tran.target, tempTime, test_index + 1, mut_tran)
                else:
                    tree_create(tran.target, tempTime, test_index + 1, [])
                #tree_create(tran.target, tempTime, test_index + 1, tran_dict[tran.target])

    tree_create(muts_NFA.init_state, 0, 0, [])
    return C_test, C


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
                if tran.target == muts_NFA.sink_state:
                    continue
                tree_create(tran.target, tempTime, test_index + 1)

    tree_create(muts_NFA.init_state, 0, 0)
    return C_test, C


# transition mutation
def mutation_tran(hypothesis, k, region_num, upper_guard, tests):
    Tsel = []
    # 生成变异体
    mutations, new_trans = tran_mutation_generation(hypothesis, k, region_num, upper_guard)
    print('number of tran_mutations', len(mutations))
    # 生成NFA
    #muts_NFA = NFA_generation(mutations, hypothesis)
    muts_NFA = NFA_generation_tran(new_trans, hypothesis)
    print('number of tran NFA trans', len(muts_NFA.trans))
    # 变异分析
    print('Starting mutation analysis...')
    tran_dict = get_tran_dict(muts_NFA)
    tests_valid = []
    C = []
    C_tests = []
    for test in tests:
        C_test, C = mutation_analysis_tran(muts_NFA, hypothesis, test, C, tran_dict)
        if C_test:
            tests_valid.append(test)
            C_tests.append(C_test)
    if C:
        cover_rate = float(len(C)) / float(len(new_trans))
        print("tests cover rate:", cover_rate)
    # 测试筛选
    if len(C_tests) > 10:
        Tsel = test_selection(tests_valid, C, C_tests)
    else:
        Tsel = tests_valid
    return Tsel


# tran mutation generation
def tran_mutation_generation(hypothesis, k, region_num, upper_guard):
    mutations = []
    new_trans = []
    step_trans_dict = {}
    mut_num = 0
    for state in hypothesis.states:
        step_trans_dict[state] = k_step_trans(hypothesis, state, k)
    for tran in hypothesis.trans:
        if tran.source == hypothesis.sink_state and tran.target == hypothesis.sink_state:
            continue
        trans = split_tran_guard_remove_first(tran, region_num, upper_guard)
        for state in hypothesis.states:
            #if random.random() > 0.5:
            #    continue
            if state != tran.target:
                for prefix in trans:
                    temp = deepcopy(prefix)
                    temp.target = state
                    temp.tran_id = 'tran' + str(mut_num)
                    mut_num += 1
                    new_trans.append(temp)
                    #for suffix in step_trans_dict[state]:
                    #    mutations.append([temp] + suffix)
    return mutations, new_trans


# 测试筛选
def test_selection(Tests, C, C_tests):
    Tsel = []
    c = deepcopy(C)  # all mutations
    tests = deepcopy(Tests)  # tests
    cset = deepcopy(C_tests)  # tests 对应的 cover mutation set
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
                recursion(tran.target, deepcopy(paths) + [tran])

    recursion(q, [])
    return trans_list


# 将Model中的迁移按source分组
def get_tran_dict(Model):
    tran_dict = {}
    for tran in Model.trans:
        if tran.source in tran_dict.keys():
            tran_dict[tran.source].append(tran)
        else:
            tran_dict[tran.source] = [tran]
    return tran_dict


# get mutated access seq leading to a single state
def get_all_acc(hypothesis, state, state_num):
    paths = []
    max_path_length = min(int(len(hypothesis.states) * 1.5), state_num * 1.5)
    if state == hypothesis.init_state:
        paths.append([])

    def get_next_tran(sn, path):
        if len(path) > max_path_length or sn == hypothesis.sink_state:
            return True
        if sn == state and path:
            if path not in paths:
                paths.append(path)
        for tran in hypothesis.trans:
            if tran.source == sn:
                if len(path) > 0 and tran == path[-1]:
                    continue
                get_next_tran(tran.target, deepcopy(path) + [tran])

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


# 将迁移的guard分割 - 不取切分的第一个???
def split_tran_guard_remove_first(tran, region_num, upper_guard):
    trans = []
    for guard in tran.guards:
        temp_guards = guard_split(guard, region_num, upper_guard)
        if not temp_guards:
            trans.append(OTATran('', tran.source, tran.action, [guard], tran.reset, tran.target))
            trans.append(OTATran('', tran.source, tran.action, [guard], not tran.reset, tran.target))
        for temp_guard in temp_guards:
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], tran.reset, tran.target))
            trans.append(OTATran('', tran.source, tran.action, [temp_guard], not tran.reset, tran.target))
    return trans

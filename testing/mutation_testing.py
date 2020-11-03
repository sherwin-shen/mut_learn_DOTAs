import random
import copy
import math
from common.TimedWord import TimedWord
from testing.random_testing import test_generation_2
from common.hypothesis import OTATran, get_split_guards
from common.TimeInterval import Guard


class Mut(object):
    def __init__(self, hypothesis, location_pre, action_pre, v_actions, mId):
        self.h = hypothesis
        self.location_pre = location_pre
        self.action_pre = action_pre
        self.v_actions = v_actions
        self.mId = mId

    def show(self):
        print("mId: " + str(self.mId))
        print("location_pre: " + str(self.location_pre))
        print("action_pre: " + str(self.action_pre))
        print("v_actions: " + str(self.v_actions))


class NFAMut(object):
    def __init__(self, states, s_state, inputs, trans, f_states):
        self.states = states
        self.s_state = s_state
        self.inputs = inputs
        self.trans = trans
        self.f_states = f_states


# 变异测试主函数
def mutation_testing(hypothesisOTA, upper_guard, state_num, minimal_duration, system):
    equivalent = True
    ctx = None

    # 参数配置
    pretry = 0.9
    pstop = 0.02
    max_steps = min(int(2 * state_num), int(2 * len(hypothesisOTA.states)))
    linfix = min(math.ceil(len(hypothesisOTA.states) / 2), math.ceil(state_num / 2))
    test_num = int(len(hypothesisOTA.states) * len(hypothesisOTA.actions) * upper_guard * 10)
    nsel = 2000

    # 测试集生成
    tests = []
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    # guard 变异分析
    guard_tests = mutation_guard(hypothesisOTA, minimal_duration, upper_guard, tests, nsel)
    # 测试执行
    if len(guard_tests) > 0:
        print('guard mutation', len(guard_tests))
        equivalent, ctx = test_execution(hypothesisOTA, system, guard_tests)

    # 如果未找到反例
    if equivalent:
        # state 变异分析
        state_tests = mutation_state(hypothesisOTA, tests, nsel, guard_tests, state_num, minimal_duration, upper_guard)
        # 测试执行
        if len(state_tests) > 0:
            print('state mutation', len(state_tests))
            equivalent, ctx = test_execution(hypothesisOTA, system, state_tests)

    # # 当找不到反例的时候检测当前测试集是否找不到
    # if equivalent:
    #     temp_equivalent, temp_ctx = test_execution(hypothesisOTA, system, tests)
    #     print(temp_equivalent, [i.show() for i in temp_ctx])
    return equivalent, ctx


# guard 变异
def mutation_guard(hypothesis, minimal_duration, upper_guard, tests, nsel):
    Tsel = []
    # 生成变异体
    mutations = mutant_generation_guard(hypothesis, minimal_duration, upper_guard)  # 生成的是tran list
    # 生成NFA
    NFA_mut = NFAMut(hypothesis.states, hypothesis.init_state, hypothesis.actions, hypothesis.trans + mutations, [])
    # 变异分析 - pre_tests 与 cMuts 分别一一对应
    NFA_mut_tran_dict = get_tran_dict_guard(NFA_mut)
    pre_tests = []
    cMuts = []
    IMutsel = []
    for test in tests:
        cMut, IMutsel = mutation_analysis_guard(test, NFA_mut, IMutsel, NFA_mut_tran_dict)
        if cMut:
            pre_tests.append(test)
            cMuts.append(cMut)
    # 测试筛选
    if cMuts:
        Tsel = test_selection([], pre_tests, IMutsel, cMuts, nsel)
    return Tsel


# 生成变异体 - guard 算子
def mutant_generation_guard(hypothesis, step, upper_guard):
    new_trans = []
    tran_id = 0
    for tran in hypothesis.trans:
        for guard in tran.guards:
            guard_min = guard.get_min()
            guard_max = guard.get_max()
            # 特殊情况处理 - [0,+)
            if guard_min == 0 and guard.get_closed_min() and guard_max == float("inf"):
                temp_guards = get_split_guards(guard, step, upper_guard)
                for state in hypothesis.states:
                    for temp_guard in temp_guards:
                        if state == tran.target:
                            continue
                        new_trans.append(OTATran('new_' + str(tran_id), tran.source, tran.action, [temp_guard], tran.reset, state))
                        tran_id += 1
                continue
            # 正常情况处理 - 处理左边
            if guard_min == 0:
                if not guard.get_closed_min():
                    new_trans.append(OTATran("new" + str(tran_id), tran.source, tran.action, [Guard("[0,0]")], tran.reset, tran.target))
                    tran_id += 1
            else:
                if guard.get_closed_min():
                    left_guard = Guard('[0,' + str(guard_min) + ')')
                else:
                    left_guard = Guard('[0,' + str(guard_min) + ']')
                temp_guards = get_split_guards(left_guard, step, upper_guard)
                for temp_guard in temp_guards:
                    new_trans.append(OTATran("new" + str(tran_id), tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    tran_id += 1
            # 正常情况处理 - 处理右边
            if guard_max == float("inf") or guard_max > upper_guard:
                pass
            elif guard_max == upper_guard:
                pass
            else:
                if guard.get_closed_max():
                    left_guard = Guard('(' + str(guard_max) + ',+)')
                else:
                    left_guard = Guard('[' + str(guard_max) + ',+)')
                temp_guards = get_split_guards(left_guard, step, upper_guard)
                for temp_guard in temp_guards:
                    new_trans.append(OTATran("new" + str(tran_id), tran.source, tran.action, [temp_guard], tran.reset, tran.target))
                    tran_id += 1
    return new_trans


# guard变异分析 - 获得test覆盖到的mut
def mutation_analysis_guard(test, NMut, IMutsel, NFA_mut_guard_tran_dict):
    cMut = []
    catch = {}
    for i in range(len(test)):
        catch[i] = []

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return True
        cur_time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, cur_time)
        cur_trans = NFA_mut_guard_tran_dict[state]
        for tran in cur_trans:
            if tran.is_passing_tran(new_LTW):
                if tran.reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if isinstance(tran.tran_id, str):
                    if tran.tran_id not in cMut:
                        cMut.append(tran.tran_id)
                    if tran.tran_id not in IMutsel:
                        IMutsel.append(tran.tran_id)
                if [tran.target, tempTime] not in catch[test_index]:
                    catch[test_index].append([tran.target, tempTime])
                    tree_create(tran.target, tempTime, test_index + 1)

    tree_create(NMut.s_state, 0, 0)
    return cMut, IMutsel


# state 变异
def mutation_state(hypothesis, tests, nsel, guard_tests, state_num, minimal_duration, upper_guard):
    hypothesis_guard_split = hypothesis.guard_split(minimal_duration, upper_guard)
    Tsel = []
    # 生成变异体
    nacc = 10
    mutations = mutant_generation_state(hypothesis_guard_split, nacc, state_num)
    # 生成NFA
    # # mut由于数量较大，因此可进行一定的筛选
    # IMut_sample = mutant_sample(hypothesisOTA.states, mutations)
    IMut_sample = mutations
    NFA_mut = NFA_mutant_state(hypothesis_guard_split, IMut_sample)
    # 变异分析 - pre_tests 与 cMuts 分别一一对应
    NFA_mut_tran_dict = get_tran_dict(NFA_mut)
    pre_tests = []
    cMuts = []
    IMutsel = []
    for test in tests:
        cMut, IMutsel = mutation_analysis_state(test, NFA_mut, IMutsel, NFA_mut_tran_dict)
        if cMut:
            pre_tests.append(test)
            cMuts.append(cMut)
    # 测试筛选
    if cMuts:
        Tsel = test_selection(guard_tests, pre_tests, IMutsel, cMuts, nsel)
    return Tsel


# 生成变异体 - state split 算子
def mutant_generation_state(hypothesis, nacc, state_num):
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
                    tMut, mId = mut_split(s1, s2, hypothesis, mId)
                    if tMut is not None:
                        Muts.extend(tMut)
    return Muts


# 变异体筛选
def mutant_sample(states, mutations):
    pass


# 生成NFA - state 变异
def NFA_mutant_state(hypothesis, IMuts):
    states = []
    s_state = [hypothesis.init_state]
    inputs = hypothesis.actions
    trans = []
    f_states = []

    for s in hypothesis.states:
        states.append([s])
    for ts in hypothesis.trans:
        trans.append([[ts.source], ts, [ts.target]])
    x = 0
    for mut in IMuts:
        s = [x, mut.mId]
        states.append(s)
        x += 1
        trans.append([[mut.location_pre], mut.action_pre, s])
        v = mut.v_actions
        j = 0
        while j <= len(v) - 1:
            ss = [x, mut.mId]
            states.append(ss)
            x += 1
            trans.append([s, v[j], ss])
            if j == len(v) - 1:
                if ss not in f_states:
                    f_states.append(ss)
            s = ss
            j += 1
    return NFAMut(states, s_state, inputs, trans, f_states)


# state变异分析 - 获得test覆盖到的mut
def mutation_analysis_state(test, NMut, IMutsel, NFA_mut_tran_dict):
    F_states = NMut.f_states
    cMut = []

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return True
        cur_time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, cur_time)
        if str(state) not in NFA_mut_tran_dict.keys():
            return True
        cur_trans = NFA_mut_tran_dict[str(state)]
        for tran in cur_trans:
            if tran[1].is_passing_tran(new_LTW):
                if tran[1].reset:
                    tempTime = 0
                else:
                    tempTime = cur_time
                if len(tran[2]) == 2:
                    if tran[2] in F_states and (tran[2][1] not in cMut):
                        cMut.append(tran[2][1])
                        if tran[2][1] not in IMutsel:
                            IMutsel.append(tran[2][1])
                tree_create(tran[2], tempTime, test_index + 1)
        return True

    tree_create(NMut.s_state, 0, 0)
    return cMut, IMutsel


# 测试筛选
def test_selection(cache_tests, Tests, C, Cset, nsel):
    Tsel = []
    CC = copy.deepcopy(C)  # all mutations
    tests = copy.deepcopy(Tests)  # tests
    cset = copy.deepcopy(Cset)  # tests 对应的 cover mutation set
    pre_set = []
    while len(Tsel) < nsel and CC:
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

# 将NFA中的迁移按source state分组
def get_tran_dict_guard(NFA_mut):
    tran_dict = {}
    for tran in NFA_mut.trans:
        if tran.source in tran_dict.keys():
            tran_dict[tran.source].append(tran)
        else:
            tran_dict[tran.source] = [tran]
    return tran_dict


# 将NFA中的迁移按source state分组
def get_tran_dict(NFA_mut):
    tran_dict = {}
    for tran in NFA_mut.trans:
        if str(tran[0]) in tran_dict.keys():
            tran_dict[str(tran[0])].append(tran)
        else:
            tran_dict[str(tran[0])] = [tran]
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
            return True
        for tran in hypothesis.trans:
            if tran.source == sn:
                if tran.target == sn and len(path) >= 2:
                    if path[-1].source == sn and path[-2].source == sn:
                        continue
                get_next_tran(tran.target, copy.deepcopy(path) + [tran])

    get_next_tran(hypothesis.init_state, [])
    return paths


# 状态分割生成变异体
def mut_split(s1, s2, hypothesis, mId):
    if len(s1) < len(s2) and s2[0:len(s1)] == s1:
        return None, mId
    sqSuf = arg_maxs(s1, s2)
    sqSuf_tran = []
    for sq in sqSuf:
        sqSuf_tran.append(sq)
    ss1 = s1[0:len(s1) - len(sqSuf_tran)]
    if len(ss1) == 0:
        return None, mId
    pI = ss1[len(ss1) - 1]
    Mutants = list()
    step = 1
    Ik = get_Ik(hypothesis, s1[-1].target, step)
    for distSeq in Ik:
        H = hypothesis
        qpre = pI.source
        v = sqSuf_tran + distSeq
        M1 = Mut(H, qpre, pI, v, mId)
        mId += 1
        Mutants.append(M1)
    return Mutants, mId


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
def get_Ik(hypothesis, qs, step):
    IK = []

    def recursion(cur_state, paths):
        if len(paths) == step:
            if paths not in IK:
                IK.append(paths)
            return True
        for tran in hypothesis.trans:
            if tran.source == cur_state:
                recursion(tran.target, copy.deepcopy(paths) + [tran])

    recursion(qs, [])
    return IK


# 在Tests中找到能够覆盖C中最多的测试集
def arg_min(Tests, C, Cset):  # C: mutations sets; Cset: cover sets
    min_num = len(C)
    index = 0
    i = 0
    for ct in Cset:
        if not ct:
            i += 1
            continue
        CC = copy.deepcopy(C)
        for c in ct:
            if c in CC:
                CC.remove(c)
        t_min = len(CC)
        if t_min < min_num:
            index = i
            min_num = t_min
        i += 1
    return index, Tests[index]


# prefix set of tws （tws前缀集）
def prefixes(tws):
    new_prefixes = []
    for i in range(1, len(tws) + 1):
        temp_tws = tws[:i]
        new_prefixes.append(temp_tws)
    return new_prefixes

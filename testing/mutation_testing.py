import random
import copy
from common.TimedWord import TimedWord
from testing.random_testing import test_generation_2


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


# --------------------------------- 算法1 - mutation testing用于测试集筛选 ---------------------------------

def mutation_testing_1(hypothesisOTA, upper_guard, system):
    # 参数配置
    hypothesisOTA = hypothesisOTA.build_simple_hypothesis()
    test_num = len(hypothesisOTA.states) * len(hypothesisOTA.actions) * 200
    pretry = 0.9
    pstop = 0.02
    linfix = int(len(hypothesisOTA.states) / 2) + 1
    max_steps = int(1.5 * len(hypothesisOTA.states))

    # 生成候选测试集
    tests = []
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    # 特殊处理状态为1情况
    if len(hypothesisOTA.states) == 1:
        equivalent, ctx = test_execution(hypothesisOTA, system, tests)
        return equivalent, ctx

    # 生成变异体
    nacc = 10
    mutations = mutant_generation_state(hypothesisOTA, nacc)

    # 变异分析生成测试集
    Tsel = []  # mutation所筛选出来的测试集
    nsel = 2000
    if len(mutations) > 0:
        # # mut由于数量较大，因此可进行一定的筛选
        # IMut_sample = mutant_sample(hypothesisOTA.states, mutations)
        IMut_sample = mutations
        NFA_mut = NFA_mutant(hypothesisOTA, IMut_sample)  # 生成包含muts信息的NFA

        # 找到test覆盖的muts，以下数组一一对应
        pre_tests = []
        cMuts = []
        IMutsel = []
        for test in tests:
            cMut, IMutsel = mutation_analysis(test, NFA_mut, IMutsel)
            if cMut:
                pre_tests.append(test)
                cMuts.append(cMut)
        if cMuts:
            Tsel = test_selection(pre_tests, IMutsel, cMuts, nsel)
    else:
        raise Exception("Mutation Failed!")

    # 测试执行
    equivalent = True
    ctx = None
    if len(Tsel) > 0:
        equivalent, ctx = test_execution(hypothesisOTA, system, Tsel)
    else:
        pass
    return equivalent, ctx


# 根据状态分裂来生成
def mutant_generation_state(hypothesis, nacc):
    Muts = []
    mId = 0
    for state in hypothesis.states:
        set_accq = get_all_acc(hypothesis, state)
        if len(set_accq) <= 1:
            continue
        elif nacc >= len(set_accq):
            subset_accq = set_accq
        else:
            subset_accq = random.sample(set_accq, nacc)
        for s1 in subset_accq:
            if len(s1) < 1:
                continue
            for s2 in subset_accq:
                if len(s2) < 1:
                    continue
                if s1 == s2:
                    continue
                elif len(s1) < len(s2) and s2[0:len(s1)] == s1:
                    continue
                else:
                    tMut, mId = mut_split(s1, s2, hypothesis, mId)
                    if tMut is None:
                        continue
                    Muts.extend(tMut)
    return Muts


# 变异体筛选
def mutant_sample(states, mutations):
    pass


# 根据muts生成NFA
def NFA_mutant(hypothesis, IMuts):
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


# 变异分析 - 获得test覆盖到的mut
def mutation_analysis(test, NMut, IMutsel):
    Trans = NMut.trans
    F_states = NMut.f_states
    cMut = []

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return
        time = test[test_index].time + preTime
        new_LTW = TimedWord(test[test_index].action, time)
        if len(state) == 2:
            if state in F_states and (state[1] not in cMut):
                cMut.append(state[1])
                if state[1] not in IMutsel:
                    IMutsel.append(state[1])
        for tran in Trans:
            if tran[0] == state and tran[1].is_passing_tran(new_LTW):
                if tran[1].reset:
                    tempTime = 0
                else:
                    tempTime = time
                tree_create(tran[2], tempTime, test_index + 1)

    tree_create(NMut.s_state, 0, 0)
    return cMut, IMutsel


# 测试选择
def test_selection(Tests, C, Cset, nsel):
    # Tests为当前所有tests, C为所有mutations, Cset为cover mutation set, nsel为选择的最大数量
    Tsel = []
    CC = copy.deepcopy(C)
    tests = copy.deepcopy(Tests)
    cset = copy.deepcopy(Cset)
    while len(Tsel) < nsel and CC:
        index, topt = arg_min(tests, CC, cset)
        Ctopt = cset[index]
        if not Ctopt:
            break
        intersectionEmpty = True
        for c in Ctopt:
            if c in CC:
                intersectionEmpty = False
                break
        if intersectionEmpty:
            break
        Tsel.append(topt)
        for c in Ctopt:
            if c in CC:
                CC.remove(c)
    return Tsel


# get mutated access seq leading to a single state
def get_all_acc(hypothesis, state):
    paths = []
    max_path_length = int(len(hypothesis.states) * 2.5)

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
    step = 2
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


# 找到 qs状态后走step的所有路径
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


# --------------------------------- 算法2 - mutation testing用于测试集生成 ---------------------------------

def mutation_testing_2(hypothesisOTA, upper_guard, system):
    pass


# --------------------------------- auxiliary function ---------------------------------

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

import random
import queue
import copy
from common.TimedWord import TimedWord
from common.TimeInterval import Guard
from common.hypothesis import OTATran
from testing.random_testing import test_generation_2


# --------------------------------- 算法1 - mutation testing用于测试集筛选 ---------------------------------

def mutation_testing_2(hypothesisOTA, upper_guard, state_num, system):
    # 生成候选测试集
    test_num = len(hypothesisOTA.states)*len(hypothesisOTA.states)*upper_guard*10
    tests = []
    pretry = 0.9
    pstop = 0.02
    linfix = int(len(hypothesisOTA.states) / 2)+1
    max_steps = int(1.5 * state_num)
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))
    print("lenth of tests:", len(tests))

    #状态数为1 特殊处理：
    if len(hypothesisOTA.states) == 1:
        equivalent, ctx = mutant_generation_guard(hypothesisOTA, system, tests)
        return equivalent, ctx


    # 生成变异体及变异分析
    Tsel = []
    cMuts = []
    IMutsel = []
    nacc = 10
    nsel = 100
    mutations = mutant_generation_split(hypothesisOTA, nacc)
    if len(mutations) > 0:
        #IMutsample = mutant_sample(hypothesisOTA.states, mutations)
        IMutsample = mutations
        NMut = NFA_mutant(hypothesisOTA, IMutsample)
        pre_tests = []
        for t0 in tests:
            cMut, IMutsel = mutation_analysis(t0, NMut, IMutsel)
            if cMut:
                pre_tests.append(t0)
                cMuts.append(cMut)

        print("lenth of cMuts:", len(cMuts))
        print("lenth of IMutsel:", len(IMutsel))
        print("lenth of pre_tests:", len(pre_tests))
        if cMuts:
            Tsel = test_selection(pre_tests, IMutsel, cMuts, 2000)  # 选择尽可能少的测试集覆盖cMuts里所包含的muts
        print("lenth of Tsel:", len(Tsel))
    else:
        #raise Exception("Mutation Failed!")
        print("Mutation Failed!")
        equivalent, ctx = mutant_generation_guard(hypothesisOTA, system, tests)

    # 测试执行
    if len(Tsel) > 0:
        equivalent, ctx = test_execution(hypothesisOTA, system, Tsel)

    #else:
        #raise Exception("Mutation Failed!")
    return equivalent, ctx

def mutation_testing_1(hypothesisOTA, upper_guard, state_num, system):
    # 生成候选测试集
    test_num = len(hypothesisOTA.states)*len(hypothesisOTA.states)*upper_guard*10
    tests = []
    pretry = 0.9
    pstop = 0.02
    linfix = int(len(hypothesisOTA.states) / 2)+1
    max_steps = int(1.5 * state_num)
    for i in range(test_num):
        tests.append(test_generation_2(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))
    print("lenth of tests:", len(tests))

    #状态数为1 特殊处理：
    #if len(hypothesisOTA.states) == 1:
    #    equivalent, ctx = mutant_generation_guard(hypothesisOTA, system, tests)
    #    return equivalent, ctx


    # 生成变异体及变异分析
    Tsel = []
    cMuts = []
    IMutsel = []
    nacc = 10
    nsel = 100
    equivalent, ctx = mutant_generation_guard(hypothesisOTA, system, tests)

    #else:
        #raise Exception("Mutation Failed!")
    return equivalent, ctx


class Mut(object):
    def __init__(self, hypothesis, location_pre, action_pre, v_actions, mId):
        self.h = hypothesis
        self.location_pre = location_pre
        self.action_pre = action_pre
        self.v_actions = v_actions
        self.mId = mId

    def showMut(self):
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

'''
class CoverATree_Node(object):
    def __init__(self,state,nodetime):
        self.state = state
        self.nodeTime = nodetime
        self.nextstates = []

    def addNextStates(self,states):
        self.nextstates.append(states)
'''


def mutant_generation_guard(hypothesis, system, Tests):
    cMuts = []
    IMutsel = []
    mId=0
    #next_trans = list(len(hypothesis.states))
    next_trans = {}
    #pre_trans = list(len(hypothesis.states))
    for state in hypothesis.states:
        next_trans[state] = []

    for tran in hypothesis.trans:
        next_trans[tran.source].append(tran)
        #pre_trans[int(tran.target)].append(tran)
    tran_id = 0
    new_trans = get_guardshift_trans(hypothesis, tran_id, next_trans)
    NFA_mut_guard = NFAMut(hypothesis.states, hypothesis.init_state, hypothesis.actions, hypothesis.trans + new_trans, [])

    print("lenth of guard new_trans:", len(new_trans))

    pre_tests = []
    Tsel = []
    for test in Tests:
        cMut, IMutsel = mutation_analysis_guard(test, NFA_mut_guard, IMutsel, len(new_trans))
        if cMut:
            pre_tests.append(test)
            cMuts.append(cMut)
    if cMuts:
        Tsel = test_selection(pre_tests, IMutsel, cMuts, 2000)  # 选择尽可能少的测试集覆盖cMuts里所包含的muts

    print("lenth of guard cMuts:", len(cMuts))
    print("lenth of guard IMutsel:", len(IMutsel))
    print("lenth of guard pre_tests:", len(pre_tests))
    if Tsel :
        equivalent, ctx = test_execution(hypothesis, system, Tsel)
    print("lenth of guard Tsel:", len(Tsel))

    return equivalent, ctx

def mutation_analysis_guard(test, NMut, IMutsel, max_num):
    s0 = NMut.s_state
    Trans = NMut.trans
    #F_states = NMut.f_states
    cMut = []
    j = 0
    nowTime = 0

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return
        if len(IMutsel) >= max_num:
            return
        time = test[test_index].time + preTime
        newTest = TimedWord(test[test_index].action, time)

        #if preTran.tran_id[0:3] == "new":
        #    IMutsel.append(preTran.tran_id)

        for tran in Trans:
            if tran.source == state and tran.is_passing_tran(newTest):
                if tran.reset:
                    tempTime = 0
                else:
                    tempTime = time
                #if tran.tran_id[0:3] == "new":
                if isinstance(tran.tran_id, str):
                    if tran.tran_id not in cMut:
                        cMut.append(tran.tran_id)
                    if tran.tran_id not in IMutsel:
                        IMutsel.append(tran.tran_id)
                tree_create(tran.target, tempTime, test_index + 1)
    tree_create(s0, nowTime, j)
    return cMut, IMutsel

def get_guardshift_trans(hypothesis, tran_id, next_trans):
    new_trans = []
    for q in hypothesis.states:
        for tran in next_trans[q]:
            for action in hypothesis.actions:
                new_guards = []
                if tran.action == action:
                    temp_value = []
                    temp_gurds = []
                    for guard in tran.guards:
                        max = guard.get_max()
                        min = guard.get_min()
                        temp_gurds.append(max)
                        temp_gurds.append(min)
                        #shift = random.randint()
                        if guard.get_closed_max():
                            temp_value.append(int(max))
                        else:
                            temp_value.append(max - 0.1)
                        if guard.get_closed_max():
                            temp_value.append(int(min))
                        else:
                            temp_value.append(min + 0.1)
                    #temp_gurds.sort()
                    temp_value.sort()
                    index1 = 0
                    #if len(temp_value) == 2:
                    #    new_guard = "[0," + str(random.randint(0, 20)) + "]"
                    while index1 < len(temp_value):
                        if index1 == 0 and int(temp_value[index1]) == 0:
                            #index1 += 1
                            if temp_value[index1 + 1] == float("inf"):
                                temp_value[0] = random.randint(0, 10)
                                temp_value[1] = random.randint(40, 60)
                                break
                            else:
                                index1 += 1
                            continue
                        elif index1 == 0 and int(temp_value[index1]) != 0:
                            pre = 0
                        else:
                            pre = temp_value[index1 - 1]
                        if index1 == len(temp_value) - 1 and temp_value[index1] != float("inf"):
                            #suf = float("inf")
                            suf = 500
                        elif index1 == len(temp_value) - 1 and temp_value[index1] == float("inf"):
                            #suf = 500
                            temp_value[index1] = temp_value[index1 -1] + 100
                            suf = temp_value[index1] + 100
                        else:
                            suf = temp_value[index1 + 1]
                        if coin_flip(0.5):
                            shift_value = temp_value[index1] - random.randint(0, int(temp_value[index1] - pre))
                        else:
                            shift_value = temp_value[index1] + random.randint(0, int(suf - temp_value[index1]))
                        temp_value[index1] = shift_value
                        index1 += 1
                    index2 = 0
                    while index2 < len(temp_value) - 1:
                        if isinstance(temp_value[index2], int):
                            new_guard = "[" + str(temp_value[index2]) + ","
                        else:
                            new_guard = "(" + str(round(temp_value[index2])) + ","
                        if isinstance(temp_value[index2 + 1], int):
                            new_guard += str(temp_value[index2 + 1]) + "]"
                        else:
                            new_guard += str(round(temp_value[index2 + 1])) + ")"
                        new_guards.append(Guard(new_guard))
                        index2 += 1
                    new_trans.append(OTATran("new"+str(tran_id), tran.source, tran.action, new_guards, tran.reset, tran.target))
                    tran_id += 1
    return new_trans



def mutant_generation_split(hypothesis, nacc):
    Muts = []
    mId=0

    for q in hypothesis.states:
        set_accq = get_all_acc(hypothesis, q)
        if len(set_accq) < 2:
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
                else:
                    tMut, mId = mut_split(s1, s2, hypothesis, mId)
                    if tMut is None:
                        continue
                    Muts.extend(tMut)
    return Muts


def mutant_sample(States, IMut):
    IMutq = list(range(len(States)))
    IMutq_num = list(range(len(States)))
    IMuts = []
    hypothesis = IMut[0].h

    for i in range(len(States)):
        IMutq[i] = []
        IMutq_num[i] = 0
    for mut in IMut:
        for t in hypothesis.trans:
            q = t.source
            if t.source == mut.location_pre and t.tran_id == mut.action_pre.tran_id:
                q = t.target
                break
        ii = int(q)
        IMutq[ii].append(mut)
        IMutq_num[ii] += 1
    sum = int(len(IMut) / len(States))
    for muts in IMutq:
        if sum >= len(muts):
            IMuts.extend(muts)
        else:
            mut = random.sample(muts, sum)
            IMuts.extend(mut)
    return IMuts


def NFA_mutant(hypothesis, IMut):
    x = 0
    S = []
    T = list()
    F = []

    for s in hypothesis.states:
        S.append([s])

    for ts in hypothesis.trans:
        T.append([[ts.source], ts, [ts.target]])

    for M in IMut:
        s = [x, M.mId]
        S.append(s)
        x += 1
        T.append([[M.location_pre], M.action_pre, s])
        v = M.v_actions
        j = 0
        while j <= len(v) - 1:
            ss = [x, M.mId]
            S.append(ss)
            x += 1
            T.append([s, v[j], ss])
            if j == len(v) - 1:
                if ss not in F:
                    F.append(ss)
            s = ss
            j += 1
    NMut = NFAMut(S, [hypothesis.init_state], hypothesis.actions, T, F)
    return NMut

'''
def mutation_analysis1(test, NMut,IMutsel):  #abandon 弃用
    s0 = NMut.s_state
    T = NMut.trans
    F = NMut.f_states
    cMut = []
    state = [s0]
    blocked = []
    j = 0
    nowTime = 0
    while j <= len(test) - 1:
        next = []
        time = test[j].time + nowTime
        newTest = TimedWord(test[j].action, time)
        for s in state:
            n = []
            unblocked_trans = []
            ifTransBlocked = False

            for trans in T:
                if trans[0] == s and trans[1].is_passing_tran(newTest):
                    for bb in blocked:
                        if trans[0] == bb[1][0] and trans[2] == bb[1][2] and trans[1].tran_id == bb[1][1].tran_id:
                            ifTransBlocked = True
                            break
                    if not ifTransBlocked:
                        ifTransBlocked = False
                        unblocked_trans.append(trans)

                    if trans[1].reset:
                        nowTime = 0
                    else:
                        nowTime = time
            for t in unblocked_trans:
                if len(t[2]) == 2:
                    sBlockTrans = False
                    for bb in blocked:
                        if s == bb[0]:
                            sBlockTrans = True
                            if not [t[2], bb[1]] in blocked:
                                blocked.append([t[2], bb[1]])
                            if [s, bb[1]] in blocked:
                                blocked.remove([s, bb[1]])
                    if not sBlockTrans:
                        blocked.append([t[2], t])

                if t[2] not in n:
                    n.append(t[2])

            b2 = get_blocked(s, blocked)
            if len(n) == 0 and b2:
                for b in b2:
                    blocked.remove(b)
            for n1 in n:
                if n1 not in next:
                    next.append(n1)
        for sn in next:
            if len(t[2]) == 2:
                if (sn in F) and (not sn[1] in cMut):
                    cMut.append(sn[1])
                    if sn[1] not in IMutsel:
                        IMutsel.append(sn[1])
        state = next
        j += 1
    return cMut, IMutsel
'''

def mutation_analysis(test, NMut, IMutsel):
    s0 = NMut.s_state
    Trans = NMut.trans
    F_states = NMut.f_states
    cMut = []
    j = 0
    nowTime = 0

    def tree_create(state, preTime, test_index):
        if test_index >= len(test):
            return
        time = test[test_index].time + preTime
        newTest = TimedWord(test[test_index].action, time)

        if len(state) == 2:
            if state in F_states and (state[1] not in cMut):
                cMut.append(state[1])
                if state[1] not in IMutsel:
                    IMutsel.append(state[1])

        for tran in Trans:
            if tran[0] == state and tran[1].is_passing_tran(newTest):
                if tran[1].reset:
                    tempTime = 0
                else:
                    tempTime = time
                tree_create(tran[2], tempTime, test_index + 1)
    tree_create(s0, nowTime, j)
    return cMut, IMutsel

'''
def mutation_analysis1(test, NMut, IMutsel):
    s0 = NMut.s_state
    T = NMut.trans
    F = NMut.f_states
    cMut = []
    j = 0
    nowTime = 0
    tree = tree_create(s0, nowTime, T, F, cMut, IMutsel, test, j)

    return cMut, IMutsel
'''


def test_selection(Tests, C, Cset, nsel): #C:all mutations; Cset:cover mutation set
    Tsel = []
    intersectionEmpty = True
    CC = copy.deepcopy(C)
    tests = copy.deepcopy(Tests)
    cset = copy.deepcopy(Cset)
    while len(Tsel) < nsel and CC:
        i, topt = arg_min(tests, CC, cset)
        Ctopt = cset[i]
        if not Ctopt:
            break
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


# --------------------------------- 算法2 - mutation testing用于测试集生成 ---------------------------------

def mutation_testing_2(hypothesisOTA, upper_guard, state_num, system):
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
                path1=copy.deepcopy(path)
                next_to_explore.put([sn, path1.append(temp_DTW)])
    return None, init_now_time

'''
def tree_create(state, preTime, Trans, F_states, cMut, IMutsel, Test, test_index):
    nowNode = CoverATree_Node(state, preTime)

    if test_index>=len(Test):
        return nowNode
    time = Test[test_index].time + preTime
    newTest = TimedWord(Test[test_index].action, time)

    if len(state) == 2:
        if state in F_states and (state[1] not in cMut):
            cMut.append(state[1])
            if state[1] not in IMutsel:
                IMutsel.append(state[1])

    for tran in Trans:
        if tran[0] == state and tran[1].is_passing_tran(newTest):
            if tran[1].reset:
                tempTime = 0
            else:
                tempTime = time

            childNode = tree_create(tran[2], tempTime, Trans, F_states, cMut, IMutsel, Test, test_index+1)
            nowNode.addNextStates(childNode)
'''
'''
def get_next_tran(hypothesis, sn, s2, paths, max_path_length, max_paths_num, tran):
    #next_tran = None
    path=[]
    
    if sn == s2:
        print([[i.source, i.target] for i in path])
        if path not in paths:
            paths.append(path)
        return tran
    print(paths)
    if len(paths )> max_paths_num:
        return tran
    if len(path) > max_path_length:
        return tran
    for tran in hypothesis.trans:
        if tran.source == sn:
            if sn == s2:
                print([[i.source, i.target] for i in path])
                if path not in paths:
                    paths.append(path)
                return tran
            if len(path) > max_path_length:
                return tran
            path.append(get_next_tran(hypothesis, tran.target, s2, paths, max_path_length,  max_paths_num, tran))
            #if len(path)>max_path_length:
            #    break
            #get_next_tran(hypothesis, tran.target, s2, paths, path, max_path_length,  max_paths_num)


    #return paths
'''
# get mutated access seq leading to a single state
def get_all_acc(hypothesis, s2):
    paths = []
    max_path_length = int(len(hypothesis.states) * 2.5)

    def get_next_tran(sn, path):
        if sn == s2 and path:
            if path not in paths:
                paths.append(path)
            return
        if len(path) > max_path_length:
            return
        for tran in hypothesis.trans:
            if tran.source == sn:
                if len(path) >= 2 and tran.target == sn:
                    if path[-1].source == sn and path[-2].source == sn:
                        continue
                get_next_tran(tran.target, copy.deepcopy(path)+[tran])

    get_next_tran(hypothesis.init_state, [])
    return paths

# get mutated access seq leading to a single state



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
    Ik = get_Ik(hypothesis, s1[-1].target, 2)
    for distSeq in Ik:
        H = hypothesis
        qpre = pI.source
        v = sqSuf_tran + distSeq
        M1 = Mut(H, qpre, pI, v, mId)
        mId += 1
        Mutants.append(M1)
    return Mutants, mId


def get_Ik1(hypothesis, qs):
    Ik = []
    for trans1 in hypothesis.trans:
        if trans1.source == qs:
            for trans2 in hypothesis.trans:
                if trans2.source == trans1.target:
                    Ik.append([trans1] + [trans2])
    return Ik


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



def arg_maxs(s1, s2):
    ts = []
    n1 = min(len(s1), len(s2))
    for i in range(n1):
        if not s1[-1 - i].tran_id == s2[-1 - i].tran_id:
            break
        ts = s1[(- 1 - i):]
    return ts


def arg_min(Tests, C, Cset): #C:mutations sets; Cset:cover sets
    min = len(C)
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
        if t_min < min:
            index = i
            min = t_min
        i += 1
    return index, Tests[index]


def get_blocked(s, blocked):
    b = []
    for bb in blocked:
        if s == bb[0]:
            b.append(bb)
    return b

def coin_flip(p):
    return random.random() <= p

'''
def tree_create1(state, preTime, Trans, F_states, cMut, IMutsel, Test, test_index):
    nowNode = CoverATree_Node(state, preTime)

    if test_index>=len(Test):
        return nowNode
    time = Test[test_index].time + preTime
    newTest = TimedWord(Test[test_index].action, time)

    if len(state) == 2:
        if state in F_states and (state[1] not in cMut):
            cMut.append(state[1])
            if state[1] not in IMutsel:
                IMutsel.append(state[1])

    for tran in Trans:
        if tran[0] == state and tran[1].is_passing_tran(newTest):
            if tran[1].reset:
                tempTime = 0
            else:
                tempTime = time

            childNode = tree_create(tran[2], tempTime, Trans, F_states, cMut, IMutsel, Test, test_index+1)
            nowNode.addNextStates(childNode)
'''

if __name__ == '__main__':
    import json
    from common.system import build_system, build_canonicalOTA
    from common.make_pic import make_system

    model_file = '../benchmarks/3_2_10/3_2_10-1/model.json'
    with open(model_file, 'r') as json_model:
        model = json.load(json_model)
    system = build_system(model)
    system = build_canonicalOTA(system)
    make_system(system, 'result_path', '/model_target')

    #paths_temp = get_Ik(system, system.states[1], 2)
    paths_temp = get_all_acc(system, '1')
    for cur_path in paths_temp:
       print([[i.source, i.target] for i in cur_path])

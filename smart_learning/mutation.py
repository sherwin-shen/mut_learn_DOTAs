import random
import queue
from copy import deepcopy
from common.TimedWord import TimedWord


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


def mutant_generation(hypothesis, upper_guard, nacc):
    Muts = []
    num = 0
    for q in hypothesis.states:
        li = nacc
        set_accq = get_all_acc(hypothesis, q, upper_guard)
        if len(set_accq) <= 1:
            continue
        elif nacc >= len(set_accq):
            subset_accq = set_accq
        else:
            subset_accq = random.sample(set_accq, li)
        for s1 in subset_accq:
            if len(s1) < 1:
                continue
            for s2 in subset_accq:
                if len(s2) < 1:
                    continue
                if s1 == s2:
                    continue
                else:
                    tMut = mut_split(s1, s2, hypothesis)
                    if tMut is None:
                        continue
                    num += len(tMut)
                    for mut in tMut:
                        if mut not in Muts:
                            Muts.append(mut)
    return Muts


def mutant_sample(States, IMut):
    IMutq = list(range(len(States)))
    IMuts = []
    num = list(range(len(States)))
    m = 0
    hypothesis = IMut[0].h

    for i in range(len(States)):
        IMutq[i] = []
        num[i] = 0
    for mut in IMut:
        for t in hypothesis.trans:
            if t.source == mut.location_pre and t.tran_id == mut.action_pre.tran_id:
                q = t.target
                break
            else:
                q = t.source
        ii = int(q)
        IMutq[ii].append(mut)
        num[ii] += 1
    for nm in num:
        m = m + nm
    m = int(m / len(States))
    for im in IMutq:
        if m > len(im):
            IMuts.extend(im)
        else:
            tm = random.sample(im, m)
            IMuts.extend(tm)
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
        s = [x, M]
        S.append(s)
        x += 1
        T.append([[M.location_pre], M.action_pre, s])
        v = M.v_actions
        j = 0
        while j <= len(v) - 1:
            ss = [x, M]
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


def mutation_analysis(test, NMut):
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
        check!!!!
        for sn in next:
            if len(t[2]) == 2:
                if (sn in F) and (not sn[1] in cMut):
                    cMut.append(sn[1])
        state = next
        j += 1
    return cMut, test


def test_selection(Tests, C, Cset, nsel):
    Tsel = []
    lable1 = False
    CC = deepcopy(C)
    while len(Tsel) < nsel and CC:
        lable2 = False
        i, topt = arg_min(Tests, CC, Cset)
        Ctopt = Cset[i]
        if not Ctopt:
            break
        for cc1 in Ctopt:
            for cc2 in CC:
                if cc1.mId == cc2.mId:
                    lable1 = True
                    lable2 = True
                    break
            if lable1:
                lable1 = False
                break
        if not lable2:
            break
        Tsel.append(topt)
        for c1 in Ctopt:
            for c2 in CC:
                if c1.mId == c2.mId:
                    CC.remove(c2)
    return Tsel


# --------------------------------- auxiliary function ---------------------------------


def get_all_acc(hypothesis, s2, upper_guard):
    s1 = hypothesis.init_state
    next0 = queue.Queue()
    next0.put([s1, None])
    paths = []
    num = 0
    max_num = 1000
    max_paths_length = 5
    while not next0.empty() and num < max_num and len(paths) < max_paths_length:
        num += 1
        [sc, path0] = next0.get()
        if path0 is None:
            path0 = []

        for i in hypothesis.inputs:
            time = random.randint(0, upper_guard * 2 + 1)
            check!!!!
            if time % 2 == 0:
                time = time // 2
            else:
                ....
            temp_DTW = TimedWord(i, time)
            pass
            sn = sc
            p = hypothesis.trans[0]
            for ts in hypothesis.trans:
                if ts.source == sc and ts.is_passing_tran(temp):
                    if ts.target != sc:
                        sn = ts.target
                        p = ts
                        break

            if sn == s2:
                path0.append(p)
                path1 = deepcopy(path0)
                if path0 not in paths:
                    num += 1
                    paths.append(path1)
                else:
                    num += 0
                path0.pop()
                break
            path0.append(p)
            path2 = deepcopy(path0)
            next0.put([sn, path2])
            path0.pop()
    return paths


def mut_split(s1, s2, hypothesis):
    if len(s1) < len(s2) and s2[0:len(s1)] == s1:
        return None
    sqSuf = arg_maxs(s1, s2)
    sqSuf_tran = []
    mId = 0
    for sq in sqSuf:
        sqSuf_tran.append(sq)
    ss1 = s1[0:len(s1) - len(sqSuf_tran)]
    if len(ss1) == 0:
        return None
    pI = ss1[len(ss1) - 1]
    Mutants = list()
    Ik = get_Ik(hypothesis, pI.target)
    for distSeq in Ik:
        distSeq = list(distSeq)
        H = hypothesis
        qpre = pI.source
        v = sqSuf_tran + distSeq
        M1 = Mut(H, qpre, pI, v, mId)
        mId += 1
        Mutants.append(M1)
    return Mutants


def get_Ik(hypothesis, qs):
    Ik = []
    for trans1 in hypothesis.trans:
        if trans1.source == qs:
            for trans2 in hypothesis.trans:
                if trans2.source == trans1.target:
                    Ik.append([trans1] + [trans2])
    return Ik


def arg_maxs(s1, s2):
    ts = []
    n1 = min(len(s1), len(s2))
    for i in range(n1):
        if not s1[-1 - i].tranId == s2[-1 - i].tranId:
            break
        ts = s1[(n1 - 1 - i):]
    return ts


def arg_min(Tests, C, Cset):
    min = len(C)
    index1 = 0
    i = 0
    for ct in Cset:
        if not ct:
            i += 1
            continue
        CC = deepcopy(C)
        for m1 in ct:
            for m2 in CC:
                if m1.mId == m2.mId:
                    CC.remove(m2)
        t_min = len(CC)
        if t_min < min:
            index1 = i
            min = t_min
        i += 1
    return index1, Tests[index1]


def get_blocked(s, blocked):
    b = []
    for bb in blocked:
        if s == bb[0]:
            b.append(bb)
    return b

# 模型结构转换为Jie An的模型
# model equivalence checking —— the model is COTA(complete OTA), in other words, hypothesis
from common.TimeInterval import Guard
from common.TimedWord import TimedWord


class State(object):
    def __init__(self, name, init, accept, flag):
        self.name = name
        self.init = init
        self.accept = accept
        self.flag = flag

    def __eq__(self, state):
        if self.name == state.name and self.init == state.init and self.accept == state.accept and self.flag == state.flag:
            return True
        else:
            return False

    def __hash__(self):
        return hash(("STATE", self.name, self.init, self.accept, self.flag))


class OTATran(object):
    def __init__(self, tran_id, source, action, guards, reset, target, flag):
        self.tran_id = tran_id
        self.source = source
        self.action = action
        self.guards = guards
        self.reset = reset
        self.target = target
        self.flag = flag

    def is_passing_tran(self, ltw):
        if ltw.action == self.action:
            for guard in self.guards:
                if guard.is_in_interval(ltw.time):
                    return True
        else:
            return False
        return False

    def show_guards(self):
        temp = self.guards[0].show()
        for i in range(1, len(self.guards)):
            temp = temp + 'U' + self.guards[i].show()
        return temp


class OTA(object):
    def __init__(self, actions, states, trans, init_state, accept_states):
        self.actions = actions
        self.states = states
        self.trans = trans
        self.init_state = init_state
        self.accept_states = accept_states

    def max_time_value(self):
        max_time_value = 0
        for tran in self.trans:
            for c in tran.guards:
                if c.max_value == '+':
                    temp_max_value = float(c.min_value) + 1
                else:
                    temp_max_value = float(c.max_value)
                if max_time_value < temp_max_value:
                    max_time_value = temp_max_value
        return max_time_value

    def find_state_by_name(self, state_name):
        for state in self.states:
            if state.name == state_name:
                return state
        return None


class Letter(object):
    def __init__(self, state, guard):
        self.state = state
        if isinstance(guard, str):
            guard = Guard(guard)
        self.guard = guard

    def __eq__(self, letter):
        if self.state == letter.state and self.guard == letter.guard:
            return True
        else:
            return False

    def __hash__(self):
        return hash(("LETTER", self.state, self.guard))


class Letterword(object):
    def __init__(self, lw, prelw, action):
        self.lw = lw
        self.prelw = prelw
        self.action = action

    def __eq__(self, letterword):
        return self.lw == letterword.lw

    def __hash__(self):
        return hash(("LETTERWORD", self.lw, self.prelw, self.action))


# 注意，传入的model1和model2都要是complete模型
def equivalence(model1, model2, upper_guard):
    model1 = transform_model(model1, 'A')
    model2 = transform_model(model2, 'B')
    upper_guard = max(upper_guard, model1.max_time_value(), model2.max_time_value())
    res, w_pos = ota_inclusion(upper_guard, model1, model2)
    dtw_pos = None
    if not res:
        # dtw_pos is accepted by model2 but not model1
        dtw_pos = find_DTWs(w_pos, 'A')

    res2, w_neg = ota_inclusion(upper_guard, model2, model1)
    dtw_neg = None
    if not res2:
        # dtw_neg is accepted by model1 but not model2
        dtw_neg = find_DTWs(w_neg, 'B')

    if res and res2:
        return True, None

    ctx = None
    if res and not res2:
        ctx = dtw_neg
    elif res2 and not res:
        ctx = dtw_pos
    elif not res2 and not res:
        if len(find_path(w_pos)) <= len(find_path(w_neg)):
            ctx = dtw_pos
        else:
            ctx = dtw_neg
    return False, ctx


# Determine whether L(B) is a subset of L(A)
def ota_inclusion(max_time_value, A, B):
    A_init_name = A.init_state
    B_init_name = B.init_state
    L1 = A.find_state_by_name(A_init_name)
    Q1 = B.find_state_by_name(B_init_name)
    w0 = [{Letter(L1, "[0,0]"), Letter(Q1, "[0,0]")}]
    to_explore = [Letterword(w0, None, 'INIT')]
    explored = []
    while True:
        if len(to_explore) == 0:
            return True, None
        w = to_explore[0]
        del to_explore[0]
        if is_bad_letterword(w.lw, A, B):
            return False, w
        while explored_dominated(explored, w):
            if len(to_explore) == 0:
                return True, None
            w = to_explore[0]
            del to_explore[0]
            if is_bad_letterword(w.lw, A, B):
                return False, w
        wsucc, next_list = compute_wsucc(w, max_time_value, A, B)
        for nw in next_list:
            if nw not in to_explore:
                to_explore.append(nw)
        if w not in explored:
            explored.append(w)


# Determine whether a letterword is bad in case of "L(B) is a subset of L(A)"
def is_bad_letterword(letterword, A, B):
    if len(letterword) == 1:
        letter1, letter2 = list(letterword[0])
    elif len(letterword) == 2:
        letter1, letter2 = list(letterword[0])[0], list(letterword[1])[0]
    else:
        raise NotImplementedError()
    state1 = letter1.state
    state2 = letter2.state
    if state1.flag == B.states[0].flag:
        if (state1.name in B.accept_states and state2.name not in A.accept_states) or (state1.name not in B.accept_states and state2.name in A.accept_states):
            return True
        else:
            return False
    else:
        if (state2.name in B.accept_states and state1.name not in A.accept_states) or (state2.name not in B.accept_states and state1.name in A.accept_states):
            return True
        else:
            return False


def explored_dominated(explored, w):
    if len(explored) == 0:
        return False
    for v in explored:
        if letterword_dominated(v, w):
            return True
    return False


# To determine whether letterword lw1 is dominated by letterword lw2 (lw1 <= lw2)
def letterword_dominated(lw1, lw2):
    index = 0
    flag = 0
    for letters1 in lw1.lw:
        for i in range(index, len(lw2.lw)):
            if letters1.issubset(lw2.lw[i]):
                index = i + 1
                flag = flag + 1
                break
            else:
                pass
    if flag == len(lw1.lw):
        return True
    else:
        return False


# Compute the Succ of letterword.
def compute_wsucc(letterword, max_time_value, A, B):
    results = []
    last_region = Guard('(' + str(max_time_value) + ',' + '+' + ')')
    if len(letterword.lw) == 1:
        result = letterword.lw[0]
        while any(letter.guard != last_region for letter in result):
            results.append(Letterword([result], letterword, 'DELAY'))
            new_result = set()
            for letter in result:
                new_letter = Letter(letter.state, next_region(letter.guard, max_time_value))
                new_result.add(new_letter)
            result = new_result
        current_lw = Letterword([result], letterword, 'DELAY')
        if current_lw not in results:
            results.append(current_lw)
    elif len(letterword.lw) == 2:
        if len(letterword.lw[0]) != 1 and len(letterword.lw[1]) != 1:
            raise NotImplementedError()
        result = letterword.lw
        while list(result[0])[0].guard != last_region or list(result[1])[0].guard != last_region:
            results.append(Letterword(result, letterword, 'DELAY'))
            new_result = []
            l1, l2 = list(result[0])[0], list(result[1])[0]
            if l1.guard.is_point():
                new_result.append({Letter(l1.state, next_region(l1.guard, max_time_value))})
                new_result.append({l2})
            else:
                new_result.append({Letter(l2.state, next_region(l2.guard, max_time_value))})
                new_result.append({l1})
            result = new_result
        current_lw = Letterword(result, letterword, 'DELAY')
        if current_lw not in results:
            results.append(current_lw)
            new_result = Letterword([current_lw.lw[1], current_lw.lw[0]], letterword, 'DELAY')
            if new_result not in results:
                results.append(new_result)
    else:
        raise NotImplementedError()

    # Next, perform the immediate 'a' transition
    next_list = []
    for letterword in results:
        next_ws = immediate_asucc(letterword, A, B)
        for next_w in next_ws:
            if next_w not in next_list:
                next_list.append(next_w)
    return results, next_list


# Perform the immediate 'a' action in case of L(B) is a subset of L(A).
def immediate_asucc(letterword, A, B):
    results = []
    if len(letterword.lw) == 1:
        letter1, letter2 = list(letterword.lw[0])
        for action in B.actions:
            if letter1.state.flag == A.states[0].flag:
                B_letter = immediate_letter_asucc(letter2, action, B)
                A_letter = immediate_letter_asucc(letter1, action, A)
            else:
                B_letter = immediate_letter_asucc(letter1, action, B)
                A_letter = immediate_letter_asucc(letter2, action, A)
            if B_letter is not None and A_letter is not None:
                B_is_point = B_letter.guard.is_point()
                A_is_point = A_letter.guard.is_point()
                if A_is_point and B_is_point:
                    w = [{A_letter, B_letter}]
                elif A_is_point and not B_is_point:
                    w = [{A_letter}, {B_letter}]
                elif not A_is_point and B_is_point:
                    w = [{B_letter}, {A_letter}]
                else:
                    w = [{A_letter, B_letter}]
                current_lw = Letterword(w, letterword, action)
                if current_lw not in results:
                    results.append(current_lw)
    elif len(letterword.lw) == 2:
        letter1, letter2 = list(letterword.lw[0])[0], list(letterword.lw[1])[0]
        for action in B.actions:
            if letter1.state.flag == A.states[0].flag:
                B_letter = immediate_letter_asucc(letter2, action, B)
                A_letter = immediate_letter_asucc(letter1, action, A)
                if B_letter is not None and A_letter is not None:
                    B_is_point = B_letter.guard.is_point()
                    A_is_point = A_letter.guard.is_point()
                    if A_is_point and B_is_point:
                        w = [{A_letter, B_letter}]
                    elif A_is_point and not B_is_point:
                        w = [{A_letter}, {B_letter}]
                    elif not A_is_point and B_is_point:
                        w = [{B_letter}, {A_letter}]
                    else:
                        w = [{A_letter}, {B_letter}]
                    current_lw = Letterword(w, letterword, action)
                    if current_lw not in results:
                        results.append(current_lw)
            else:
                B_letter = immediate_letter_asucc(letter1, action, B)
                A_letter = immediate_letter_asucc(letter2, action, A)
                if B_letter is not None and A_letter is not None:
                    B_is_point = B_letter.guard.is_point()
                    A_is_point = A_letter.guard.is_point()
                    if A_is_point and B_is_point:
                        w = [{A_letter, B_letter}]
                    elif A_is_point and not B_is_point:
                        w = [{A_letter}, {B_letter}]
                    elif not A_is_point and B_is_point:
                        w = [{B_letter}, {A_letter}]
                    else:
                        w = [{B_letter}, {A_letter}]
                    current_lw = Letterword(w, letterword, action)
                    if current_lw not in results:
                        results.append(current_lw)
    else:
        raise NotImplementedError()
    return results


def immediate_letter_asucc(letter, action, ota):
    state_name = letter.state.name
    region = letter.guard
    for tran in ota.trans:
        if tran.source == state_name and action == tran.action and region.is_subset(tran.guards[0]):
            succ_state_name = tran.target
            succ_state = ota.find_state_by_name(succ_state_name)
            if tran.reset:
                region = Guard("[0,0]")
            if succ_state is not None:
                return Letter(succ_state, region)
    return None


# When get a letterword, find the path ends in the letterword.
def find_path(letterword):
    current_lw = letterword
    path = [current_lw]
    while current_lw.prelw is not None:
        path.insert(0, current_lw.prelw)
        current_lw = current_lw.prelw
    return path


# Given a path, return the delay timedword.
def find_DTWs(letterword, flag):
    path = find_path(letterword)
    delay_timedwords = []
    current_clock_valuation = 0
    delay_time = 0
    for letterword in path:
        if len(letterword.lw) == 1:
            letter1, letter2 = list(letterword.lw[0])
        elif len(letterword.lw) == 2:
            letter1, letter2 = list(letterword.lw[0])[0], list(letterword.lw[1])[0]
        else:
            raise NotImplementedError()
        if letter1.state.flag == flag:
            temp_region = letter1.guard
        else:
            temp_region = letter2.guard
        if letterword.action == "DELAY":
            delay_time = minimum_in_region(temp_region) - current_clock_valuation
            current_clock_valuation = minimum_in_region(temp_region)
        elif letterword.action == 'INIT':
            pass
        else:
            new_timedword = TimedWord(letterword.action, delay_time)
            delay_timedwords.append(new_timedword)
            current_clock_valuation = minimum_in_region(temp_region)
    return delay_timedwords


# Returns r_0^1 for r_0, r_1 for r_0^1, etc.
def next_region(region, max_time_value):
    if region.is_point():
        if int(region.max_value) == max_time_value:
            return Guard('(' + region.max_value + ',' + '+' + ')')
        else:
            return Guard('(' + region.max_value + ',' + str(int(region.max_value) + 1) + ')')
    else:
        if region.max_value == '+':
            return Guard('(' + region.min_value + ',' + '+' + ')')
        else:
            return Guard('[' + region.max_value + ',' + region.max_value + ']')


# Return the minimal number in the region. For [5,9], return 5; for (4,10), return 4.5 .
def minimum_in_region(guard):
    if guard.closed_min:
        return int(guard.min_value)
    else:
        return float(guard.min_value + '.5')


# transform model structure
def transform_model(model, flag):
    states = []
    for loc in model.states:
        is_init = (loc == model.init_state)
        is_accept = (loc in model.accept_states)
        states.append(State(loc, is_init, is_accept, flag))
    trans = []
    for tran in model.trans:
        trans.append(OTATran(tran.tran_id, tran.source, tran.action, tran.guards, tran.reset, tran.target, flag))
    return OTA(model.actions, states, trans, model.init_state, model.accept_states)

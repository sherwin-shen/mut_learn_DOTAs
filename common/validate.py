from copy import deepcopy
import random
from common.equivalence import equivalence
from common.system import build_canonicalOTA
from common.TimedWord import TimedWord


def validate(learned_system, system, upper_guard):
    new_system = build_canonicalOTA(deepcopy(system))
    # 比较是否等价
    correct_flag, ctx = equivalence(learned_system, new_system, upper_guard)

    # 测试通过率
    if correct_flag:
        passingRate = 1
    else:
        failNum = 0
        testNum = 20000
        for i in range(testNum):
            sample = sample_generation(learned_system.actions, upper_guard, len(learned_system.states))
            system_res, real_value = new_system.test_DTWs(sample)
            hypothesis_res, value = learned_system.test_DTWs(sample)
            if system_res != hypothesis_res:
                failNum += 1
        passingRate = (testNum - failNum) / testNum
    return correct_flag, passingRate


def sample_generation(actions, upperGuard, stateNum):
    sample = []
    length = random.randint(1, stateNum * 2)
    for i in range(length):
        action = actions[random.randint(0, len(actions) - 1)]
        time = random.randint(0, upperGuard * 2 + 1)
        if time % 2 == 0:
            time = time // 2
        else:
            time = time // 2 + 0.5
        temp = TimedWord(action, time)
        sample.append(temp)
    return sample

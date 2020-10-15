from copy import deepcopy
from common.equivalence import equivalence
from common.system import build_canonicalOTA
from testing.random_testing import test_generation_1


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
            sample = test_generation_1(learned_system.actions, upper_guard, len(learned_system.states))
            system_res, real_value = new_system.test_DTWs(sample)
            hypothesis_res, value = learned_system.test_DTWs(sample)
            if real_value == value and system_res != hypothesis_res:
                system_res, real_value = new_system.test_DTWs(sample)
                print([i.show() for i in sample], [i.show() for i in system_res], [i.show() for i in hypothesis_res], real_value, value)
            if real_value != value:
                failNum += 1
        passingRate = (testNum - failNum) / testNum
    return correct_flag, passingRate

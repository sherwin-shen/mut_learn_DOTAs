from testing.pac_testing import pac_testing
from testing.random_testing import random_testing_1, random_testing_2, random_testing_3
from testing.pac_testing import pac_testing
from testing.mutation_testing import mutation_testing_1
from common.hypothesis import struct_simpleHypothesis
from common.make_pic import make_full_hypothesis

# LRTWs + value
def TQs(LTWs, system):
    LRTWs, value = system.test_LTWs(LTWs)
    return LRTWs, value


def EQs(hypothesisOTA, upper_guard, state_num, system, hp_num):
    # # 测试1 - pac testing
    # equivalent, ctx = pac_testing(hypothesisOTA, upper_guard, state_num, system)

    # # 测试2 - 完全随机测试
    # equivalent, ctx = random_testing_1(hypothesisOTA, upper_guard, state_num, system)

    # # 测试3 - 随机测试（用于结合mutation testing）
    # equivalent, ctx = random_testing_2(hypothesisOTA, upper_guard, state_num, system)

    # # 测试4 - 随机游走测试
    # equivalent, ctx = random_testing_3(hypothesisOTA, upper_guard, state_num, system)

    # 测试5 - 随机测试结合变异测试
    hypothesisOTA_simple = struct_simpleHypothesis(hypothesisOTA)
    make_full_hypothesis(hypothesisOTA_simple, 'results/test/', 'learned_OTA_' + str(hp_num))


    equivalent, ctx = mutation_testing_1(hypothesisOTA_simple, upper_guard, state_num, system)

    system.eq_num += 1
    return equivalent, ctx

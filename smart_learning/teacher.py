from smart_learning.mutation_testing import mutation_testing


# LRTWs + value
def TQs(LTWs, system):
    LRTWs, value = system.test_LTWs(LTWs)
    return LRTWs, value


def EQs(hypothesisOTA, upper_guard, state_num, system):
    equivalent, ctx = mutation_testing(hypothesisOTA, upper_guard, state_num, system)
    system.eq_num += 1
    return equivalent, ctx

def mutation_testing(hypothesisOTA, upper_guard, state_num, system):
    # 生成候选测试集
    test_num = 2000  # 候选测试集数量
    tests = []
    pretry = 0.9
    pstop = 0.02
    linfix = int(state_num / 2)
    max_steps = 2 * state_num
    for i in range(test_num):
        tests.append(test_case_generation(hypothesisOTA, pretry, pstop, max_steps, linfix, upper_guard))

    # 生成变异体及变异分析
    Tsel = []
    cMuts = []
    nacc = 4
    k = 2
    nsel = 100
    mutations = mutant_generation(hypothesisOTA, nacc, k)
    if len(mutations) > 0:
        IMutsel = mutant_sample(hypothesisOTA.states, mutations)
        NMut = NFA_mutant(hypothesisOTA, IMutsel)
        pre_tests = []
        for t0 in tests:
            cMut, tt = mutation_analysis(t0, NMut)
            if cMut:
                pre_tests.append(tt)
                cMuts.append(cMut)
        if cMuts:
            Tsel = test_selection(pre_tests, IMutsel, cMuts, nsel)

    # 测试执行
    max_test_num = 1000
    if len(Tsel) > 0:
        equivalent, ctx = test_execution(hypothesisOTA, system, Tsel, max_test_num)
    else:
        raise Exception("Mutation Failed!")
    return equivalent, ctx


# 测试序列生成(根据hypothesis)
def test_case_generation(hypothesis, pretry, pstop, max_steps, linfix, upper_guard):
    pass


# 测试执行
def test_execution(hypothesisOTA, system, tests, max_test_num):
    flag = True
    ctx = []
    i = 1
    for test in tests:
        if i > max_test_num:
            break
        DRTWs, value = hypothesisOTA.test_DTWs(test)
        realDRTWs, realValue = system.test_DTWs(test)
        i += 1

        if realValue != value:
            flag = False
            ctx = test
            break
    return flag, ctx

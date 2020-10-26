import copy
import smart_learning.obsTable as obsTable
from common.hypothesis import struct_discreteOTA, struct_hypothesisOTA
from smart_learning.teacher import EQs


def learnOTA_smart(system, actions, upper_guard, state_num, debug_flag):
    ### init Table
    table = obsTable.initTable(actions, system)
    if debug_flag:
        print("***************** init-Table_1 is as follow *******************")
        table.show()

    ### learning start
    equivalent = False
    learned_system = None  # learned model
    table_num = 1  # number of table
    hy_num = 1

    while not equivalent:
        ### make table prepared
        prepared = table.is_prepared()
        while not prepared:
            # make closed
            closed_flag, close_move = table.is_closed()
            if not closed_flag:
                table = obsTable.make_closed(table, actions, close_move, system)
                table_num = table_num + 1
                if debug_flag:
                    print("***************** closed-Table_" + str(table_num) + " is as follow *******************")
                    table.show()

            # make consistent
            consistent_flag, consistent_add = table.is_consistent()
            if not consistent_flag:
                consistent_flag, consistent_add = table.is_consistent()
                table = obsTable.make_consistent(table, consistent_add, system)
                table_num = table_num + 1
                if table_num == 23:
                    print('attention!')
                if debug_flag:
                    print("***************** consistent-Table_" + str(table_num) + " is as follow *******************")
                    table.show()
            prepared = table.is_prepared()

        ### build hypothesis
        # Discrete OTA
        discreteOTA = struct_discreteOTA(table, actions)
        if discreteOTA is None:
            raise Exception('Attention!!!')
        if debug_flag:
            print("***************** discreteOTA_" + str(system.eq_num + 1) + " is as follow. *******************")
            discreteOTA.show_discreteOTA()
        # Hypothesis OTA
        hypothesisOTA = struct_hypothesisOTA(discreteOTA)
        if debug_flag:
            print("***************** Hypothesis_" + str(system.eq_num + 1) + " is as follow. *******************")
            hypothesisOTA.show_OTA()

        ### EQs

        equivalent, ctx = EQs(hypothesisOTA, upper_guard, state_num, system, hy_num)
        hy_num += 1

        if not equivalent:
            # show ctx
            if debug_flag:
                print("***************** counterexample is as follow. *******************")
                print([dtw.show() for dtw in ctx])
            # deal with ctx
            table = obsTable.deal_ctx(table, ctx, system)
            table_num = table_num + 1
            if debug_flag:
                print("***************** New-Table" + str(table_num) + " is as follow *******************")
                table.show()
        else:
            learned_system = copy.deepcopy(hypothesisOTA).build_simple_hypothesis()
            for e in table.E:
                if len(e) > 1:
                    raise Exception("E len > 1")

    return learned_system, system.mq_num, system.eq_num, system.test_num, table_num

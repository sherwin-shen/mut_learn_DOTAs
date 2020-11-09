from graphviz import Digraph


# 目标系统OTA - accept
def make_system(data, filePath, fileName):
    dot = Digraph()
    for state in data.states:
        if state in data.accept_states:
            dot.node(name=str(state), label=str(state), shape='doublecircle')
        else:
            dot.node(name=str(state), label=str(state))
    for tran in data.trans:
        tranLabel = " " + str(tran.action) + " " + tran.show_guards() + " " + str(tran.reset)
        dot.edge(str(tran.source), str(tran.target), tranLabel)
    newFilePath = filePath + fileName
    dot.render(newFilePath, view=True)


# 猜想OTA - accept(忽略sink状态)
def make_hypothesis(data, filePath, fileName):
    dot = Digraph()
    states = []
    for state in data.states:
        if state != data.sink_state:
            states.append(state)
    for s in states:
        if s in data.accept_states:
            dot.node(name=str(s), label=str(s), shape='doublecircle')
        else:
            dot.node(name=str(s), label=str(s))
    for tran in data.trans:
        if tran.source != data.sink_state and tran.target != data.sink_state:
            tranLabel = " " + str(tran.action) + " " + tran.show_guards() + " " + str(tran.reset)
            dot.edge(str(tran.source), str(tran.target), tranLabel)
    newFilePath = filePath + fileName
    dot.render(newFilePath, view=True)


# 猜想OTA - accept + sink
def make_full_hypothesis(data, filePath, fileName):
    dot = Digraph()
    states = data.states
    for s in states:
        if s in data.accept_states:
            dot.node(name=str(s), label=str(s), shape='doublecircle')
        elif s == data.sink_state:
            dot.node(name=str(s), label=str(s), shape='box')
        else:
            dot.node(name=str(s), label=str(s))
    for tran in data.trans:
        tranLabel = " " + str(tran.action) + " " + tran.show_guards() + " " + str(tran.reset)
        dot.edge(str(tran.source), str(tran.target), tranLabel)
    newFilePath = filePath + fileName
    dot.render(newFilePath, view=True)

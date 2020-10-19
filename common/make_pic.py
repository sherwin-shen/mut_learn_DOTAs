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


# 猜想OTA - accept(忽略self-loop)
def make_hypothesis(data, filePath, fileName):
    dot = Digraph()
    states = data.states
    for s in states:
        if s in data.accept_states:
            dot.node(name=str(s), label=str(s), shape='doublecircle')
        else:
            dot.node(name=str(s), label=str(s))
    for tran in data.trans:
        if tran.source == tran.target and not tran.reset:
            continue
        tranLabel = " " + str(tran.action) + " " + tran.show_guards() + " " + str(tran.reset)
        dot.edge(str(tran.source), str(tran.target), tranLabel)
    newFilePath = filePath + fileName
    dot.render(newFilePath, view=True)

# 猜想OTA - accept(带self-loop)
def make_full_hypothesis(data, filePath, fileName):
    dot = Digraph()
    states = data.states
    for s in states:
        if s in data.accept_states:
            dot.node(name=str(s), label=str(s), shape='doublecircle')
        else:
            dot.node(name=str(s), label=str(s))
    for tran in data.trans:
        tranLabel = " " + str(tran.action) + " " + tran.show_guards() + " " + str(tran.reset)
        dot.edge(str(tran.source), str(tran.target), tranLabel)
    newFilePath = filePath + fileName
    dot.render(newFilePath, view=True)

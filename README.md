# Learning DOTAs via mutation testing

This is a prototype for [paper: Learning Deterministic One-Clock Timed Automata via mutation testing]().

## Overview

This project contains the implementation of an equivalence testing approach for [DOTAs learning](https://github.com/Leslieaj/OTALearning), called mutation-based equivalence testing.  The basic idea/framework comes from [Learning from Faults: Mutation Testing in Active Automata Learning](https://link.springer.com/chapter/10.1007%2F978-3-319-57288-8_2). In order to apply the basic framework to the Deterministic One-Clock Timed Automata setting, we further propose several improvements, including two special mutation operators, a heuristic test case generation algorithm, and a score-based test case selection method. By setting reasonable parameters, the newly developed mutation-based equivalence testing is a viable technique for implementing equivalence oracle to find counterexamples in a DOTAs learning setup.

## Installation

The project was developed using Python3, and you only need to download the project, but there are a few prerequisites before running：

- Python3.7.* (or high)
- graphviz (used for drawing)

## Usage

### 1. Model File

If you have prepared files `model.json` , you can directly run:

```shell
$python3 main.py model.json
```

> `model.json` is a JSON file about the structure of the model. Although this is a black box learning tool, in the prototype stage, users should provide model structure files to model DOTAs to be learned.

**model.json**

```json
{
    "states": ["1", "2"],
    "inputs": ["a", "b", "c"],
    "trans": {
        "0": ["1", "a", "[3,9)", "r", "2"],
        "1": ["1", "b", "[1,5]", "r", "2"],
        "2": ["1", "c", "[0,3)", "n", "1"],
        "3": ["2", "a", "(5,+)", "n", "1"],
        "4": ["2", "b", "(7,8]", "n", "1"],
        "5": ["2", "c", "(4,+)", "r", "1"]
    },
    "initState": "1",
    "acceptStates": ["2"]
}
```

*Explanation:*

- "states": the set of the name of locations;
- "inputs": the input alphabet;
- "trans": the set of transitions in the form `id : [name of the source location, input action, guards, reset, name of the target location];`
  - "+" in a guard means INFTY;
  - "r" means resetting the clock, "n" otherwise
- "initState": the name of initial location;
- "acceptStates": the set of the name of accepting locations.

⚠️⚠️⚠️ In the process of use, you must ensure that the naming is correct and the content follows the prescribed format.

### 2. Parameter Settings

For mutation-based equivalence testing, we have set the generally applicable parameters in advance, but users can also customize the relevant parameters in the files `./test/random_testing.py` and `./test/mutation_testing.py`.

## Output

If we learn the target DOTA successfully, the final COTA will be drawn and displayed as a PDF file. And all results will be stored in a folder named `results` and a file named `result.json`.

## License

See [MIT LICENSE](./LICENSE) file.

## Contact

Please let me know if you have any questions 👉 [EnvisionShen@gmail.com](mailto:EnvisionShen@gmail.com)


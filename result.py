import json

test_type = 'select_optimization'
model_type = '5_2_10'
for i in range(10):
    #file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(i + 1) + '/3/result.json'
    file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(1) + '/' + str(1+i) + '/result.json'
    with open(file_path, 'r') as json_model:
        model = json.load(json_model)
    print(model["testNum"])
    print(model["testNumCache"])
    print(model["actionNum"])
    print(model["passingRate"])
    print(model["correct"])

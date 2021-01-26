import json

test_type = 'mutation'
model_type = '4_2_10'
for i in range(10):
    #file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(i + 1) + '/3/result.json'
    file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(i + 1) + '/10/result.json'
    with open(file_path, 'r') as json_model:
        model = json.load(json_model)
    print(model["testNum"])
    print(model["testNumCache"])
    print(model["actionNum"])
    print(model["passingRate"])
    print(model["correct"])
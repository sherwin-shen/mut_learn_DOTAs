import json

test_type = 'temp_test_select_old'
model_set = ['4_2_10', '6_2_10', '8_2_10', '10_2_10', '6_2_20', '6_2_50', '6_4_10', '6_6_10']
#model_set = ['6_2_20', '6_2_50', '6_4_10', '6_6_10']
#model_set = ['Light', 'Train']
#model_set = ['6_2_50']
#model_type = '5_2_10'
for i in range(5):
 print("================================================")
 for model_type in model_set:
  print("////////////////////////////////////////////////")
  for j in range(2):
    #file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(i + 1) + '/3/result.json'
    file_path = './results/smart_teacher/' + test_type + '/benchmarks/' + model_type + '/' + model_type + '-' + str(1+j) + '/' + str(1+i) + '/result.json'
    #file_path = './results/smart_teacher/' + test_type + '/' + model_type + '/' + str(1 + i) + '/result.json'
    with open(file_path, 'r') as json_model:
        model = json.load(json_model)
    print(model["testNum"])
    print(model["testNumCache"])
    print(model["actionNum"])
    print(model["passingRate"])
    print(model["correct"])

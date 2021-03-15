import json

files = ["case", "4_2_10", "6_2_10", "6_2_20", "6_2_30", "6_4_10", "6_6_10", "8_2_10", "10_2_10"]

res = []
for file in files:
    for i in range(3):
        items = []
        for j in range(15):
            file_path = './results/' + file + '/' + file + '-' + str(1 + i) + '/' + str(1 + j) + '/result.json'
            with open(file_path, 'r') as json_model:
                model = json.load(json_model)
            # items.append(model["testNumCache"])
            # items.append(model["actionNum"])
            # items.append(model["passingRate"])
            if model["correct"]:
                items.append(1)

        # print(min(items))
        # print(max(items))
        print(len(items))
        # print(sum(items) / 15)
        # res.append(len(items))
        # res.append(sum(items) / 15)
# print(res)

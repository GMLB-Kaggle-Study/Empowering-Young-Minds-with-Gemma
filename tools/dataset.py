def check_data_format(data:dict):
    '''
    check data format
    ------------------------
    input data must be the following format
    {
        "version": int,
        "info": dict,
        "list": list,
    }
    '''
    if "info" not in data.keys():
    	raise AssertionError(f"Not satisfy key 'info' in data: 'info' not in {data.keys()}")
    
    if "list" not in data.keys(): 
    	raise AssertionError(f"Not satisfy key 'list' in data: 'list' not in {data.keys()}")
    
    if type(data["info"]) != dict:
        raise AssertionError(f"Not satisfy value type at 'info' key in data: {dict} not {type(data['info'])}")
        
    if type(data["list"]) != list: return False

    if "ID" not in data["info"].keys(): return False

    # list type
    for l in data["list"]:
        if type(l) != dict: return False
        if "list" not in l.keys(): return False
    
    return True

def extract_data_chatbot_formmat(data:dict):
    assert check_data_format(data)
    
    data_id = data["info"]["ID"]
    data_dict = {}

    for i in range(len(data["list"])):
        sub_data = data["list"][i]["list"]

        for j in range(len(sub_data)):
            sub_data_case = sub_data[j]
            if "audio" not in sub_data_case.keys():
                continue
            
            sub_data_case_id = f"{data_id}_{i+1:02d}_{j+1:02d}"
            # if sub_data_case_id in CONST_LIST_EXCEPT: continue
            
            sub_data_case_dict = {}
            sub_data_case_dict["0"] = {"role": "user", "content": sub_data_case["항목"]}
            now_start = ""
            before_start = ""
            prev_type = "A"
            
            for k in range(len(sub_data_case["audio"])):
                # k_key = f"{sub_data_case['audio'][k]['type']}{k//2:02d}"
                now_type = sub_data_case['audio'][k]['type']
                now_start = sub_data_case['audio'][k]["start"]
                if now_start[:6] < before_start[:6]:
                    raise ValueError(f"Timeline is not correct. Check data: {data_id}/'list'/{i}/'list'/{j}/'audio'/{k}")
                if now_start in sub_data_case_dict.keys():
                    raise KeyError(f"key '{now_start}' is already in sub_data_case_dict. Check data: {data_id}/'list'/{i}/'list'/{j}/'audio'/{k}")
                if now_type == prev_type:
                    raise TypeError(f"Now type '{now_type}' is same before conversation type. Check data: {data_id}/'list'/{i}/'list'/{j}/'audio'/{k}")
                if sub_data_case['audio'][k]['text'] == "":
                    raise ValueError(f"value is empty in sub_data_case_dict. Check data: {data_id}/'list'/{i}/'list'/{j}/'audio'/{k}")
                # sub_data_case_list.append(sub_data_case['audio'][k]['text'])
                sub_data_case_dict[now_start] = {"role": "user" if now_type=="A" else "assistant", "content": sub_data_case['audio'][k]['text']}
                before_start = sub_data_case['audio'][k]["start"]
                prev_type = sub_data_case['audio'][k]['type']

            data_dict[sub_data_case_id] = sub_data_case_dict
    
    return data_dict
from datasets import Dataset


def generate_dataset(data_dict):
    doc = []
    summ = []
    for key in data_dict.keys():
        key_list = sorted(list(data_dict[key].keys()))
        if data_dict[key][key_list[-1]]['role'] == "user": key_list = key_list[:-1]
        if len(key_list) < 2: continue
        data_segmentation = [data_dict[key][x]['content'] for x in key_list]
        doc.append(data_segmentation[:-1])
        summ.append(data_segmentation[-1])
        
        
    return Dataset.from_dict({"document": doc, "summary": summ})

def generate_chat_prompts(dataset):
    output_texts = []
    for data in dataset:
        messages = []
        
        for i in range(len(data["document"])):
            messages.append({"role": "user" if i%2==0 else "assistant", "content": data["document"][i]})
        
        messages.append({"role": "assistant", "content": data["summary"]})
        chat_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(chat_message + '<EOS>')
    return output_texts
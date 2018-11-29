import os
import pickle
import re

def list_files(dir):
    files = os.listdir(dir)
    return files



def output_to_file(file, obj):
    with open(file, 'w') as f: 
        picklestring = pickle.dumps(obj)
        f.write(picklestring)

def read_file(input_file):
    with open(input_file) as f:
        content = f.read()
        content = re.sub(r"\(.*\)","", content)
        content = re.sub(r'[^\w \n\:]', '', content)
        content = re.sub(r'\d', '', content)
        return content.splitlines()

def process_file(input_file):
    content = read_file(input_file)
    name = ""
    conversations = []

    for line in content:
        elements = line.split()
        if len(elements) == 0:
            continue
        if ':' in elements[0]:
            name = re.sub(r'[^\w]', '', elements[0])
            elements.pop(0)
        conversations.append([name, elements])
    return conversations



def main():
    files = list_files("TRN")
    for file in files:
        input_file = "TRN/" + file
        output_file = "TRN_output/" + os.path.splitext(file)[0] + '.pkl'
        obj = process_file(input_file)
        output_to_file(output_file, obj)
        print(file + " parse finished.")



if __name__ == '__main__':
    main()
import json
import sys
#from finqa_utils import program_tokenization
#from finqa_utils import get_program_op_args
import collections

filename = 'C:/Users/86199/PycharmProjects/NLP/project/dataset/train.json'
all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]
def program_tokenization(original_program):#传program（divide（），divide（））进去
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)

    program.append('EOF')
    return program

def get_program_op_args(program):#传分割的program进去（即program_tokenization返回的program）
    program = program[:-1]  # remove EOF
    # check structure
    for ind, token in enumerate(program):
        if ind % 4 == 0:
            if token.strip("(") not in all_ops:
                return False, 'None'
        if (ind + 1) % 4 == 0:
            if token != ")":
                return False, "None"

    program = "|".join(program)
    steps = program.split(")")[:-1]
    program_ops = collections.defaultdict(list)

    for ind, step in enumerate(steps):
        step = step.strip()
        if len(step.split("(")) > 2:
            return False, 'None'
        op = step.split("(")[0].strip("|").strip()
        op = op + str(ind)
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        program_ops[op].append(arg1)
        program_ops[op].append(arg2)
    return True, program_ops#program_ops
                            #return 的类型是元组！

def get_concat_prog():
    with open (filename) as f:
       input_data = json.load(f)
    concat_prog_list = []
    concat_prog = ''
    for entry in input_data:
        original_program = entry['qa']['program']
        m = program_tokenization(original_program)
        n = get_program_op_args(m)
        concat_prog = ''
        for key in n[1].keys():
            concat_prog = concat_prog + key + '_'
        concat_prog_list.append(concat_prog.strip("_"))

    return concat_prog_list
result = get_concat_prog()
print(result)